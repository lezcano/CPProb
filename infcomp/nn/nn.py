import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from infcomp.nn.address import Address
from infcomp.nn.observation import Observation, ObserveEmbeddingFC, ObserveEmbeddingLSTM, ObserveEmbeddingCNN3D4C
from infcomp.data_structures import Trace, Batch
from infcomp.settings import settings

from infcomp.parse_flatbuffers import distributionfbb_prior
from infcomp.logger import Logger
import infcomp.util as util


class NN(nn.Module):
    def __init__(self, directory="./models", file_name="infcomp-nn",
                 obs_embedding_type="fc"):

        super().__init__()
        # PriorDistr -> EmbeddingVector
        self._distribution_embedding = {}

        # Set when the first trace is processed. This will probably change in the future
        self._obs_embedding_type = obs_embedding_type
        self._obs_sequences = obs_embedding_type == "lstm"
        # Cached embedding for inference. Set in `forward`
        self._observation_embedding = None
        self._previous_lstm_hidden = None
        # Cached embedding for when the NN finds a new address in some execution
        self._prev_value_embed = None

        self._embedding_dim = settings.embedding_dim
        no = Observation.num_elems_embedding()
        self._out_lstm_dim = self._embedding_dim

        # Size of the embedding of the value + the params (if any)
        n_params = Address.num_elems_embedding()
        lstm_input_dim = (n_params + no + 1)*self._embedding_dim
        lstm_depth = 2
        self._lstm = nn.LSTM(lstm_input_dim, self._out_lstm_dim, lstm_depth)
        if settings.cuda_enabled:
            self._lstm.cuda(settings.cuda_device)

        self._optimizer = torch.optim.Adam(self.parameters())

        self._dir = directory
        self.file_name = file_name

        self.on_cuda = None
        self.cuda_device_id = None

        self.logger = None
        self.apply(util.weights_init)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.logger = Logger.logger
        # Close the current logger object but keep the statistics
        self.logger._logger = None
        self.save_nn()

    def save_nn(self, filename=None):
        if filename is None:
            filename = self.file_name
        print('Updating nn on disk...')
        if not os.path.exists(self._dir):
            print("Creating directory {}".format(self._dir))
            os.mkdir(self._dir)
        torch.save(self, os.path.join(self._dir, filename))

    def forward_obs(self, batch):
        if self.training:
            example_trace = batch.example_trace()
            obs_layer = self._get_observation_layer(example_trace.observe.value)
            if self._obs_sequences:
                traces = batch.sort_traces_observation()
                obs_list = [trace.observe.value for trace in traces]
                obs_len = [len(obs) for obs in obs_list]
                obs_seq = util.pad_sequence(obs_list, batch_first=True, variables=False)
                # Batch x Length x Dim (= 1)
                obs_seq = obs_seq.unsqueeze(2)
                obs_variable = nn.utils.rnn.pack_padded_sequence(obs_seq, obs_len, batch_first=True)
            else:
                obs_list = [trace.observe.value for trace in batch.traces]
                obs_variable = Variable(torch.stack(obs_list))
            return obs_layer(obs_variable)
        else:
            return self._observation_embedding

    def forward_prev_value(self, previous_sample, len_subbatch):
        if previous_sample is None:
            return Variable(self._default_value_embed(len_subbatch))
        else:
            try:
                prev_layer = self._get_layer(previous_sample)
                # unsqueeze(1) so that we can torch.cat it to the observation
                return prev_layer.forward_value([previous_sample.distr_fbb]).unsqueeze(1)
            except ValueError:
                return self._prev_value_embed

    def forward_subbatches(self, batch, obs_embed, previous_sample):
        embed_list = []
        indices = []
        for subbatch in batch.subbatches:
            example_trace = subbatch.example_trace()
            n_samples = len(example_trace)
            n_subbatch = len(subbatch)
            indices_subbatch = batch.get_indices(subbatch)
            indices.append(indices_subbatch)
            obs_embed_subbatch = obs_embed[indices_subbatch].unsqueeze(1)

            # Previous value
            prev_value_embed = self.forward_prev_value(previous_sample, len(subbatch))

            # Parse Distributions
            n_samples_unpack = n_samples if settings.extended_embed else n_samples - 1
            distributions_fbb = [[trace.samples[step].distr_fbb for trace in subbatch.traces] for step in range(n_samples_unpack)]
            # Get Layers
            try:
                layers = [self._get_layer(sample) for sample in example_trace.samples[:n_samples_unpack]]
            except ValueError:
                self._prev_value_embed = prev_value_embed
                raise

            # Embed Values (& maybe params as well)
            values_embeds = [prev_value_embed]
            values_embeds.extend(layer.forward_value(distr_fbb).unsqueeze(1)
                                 for layer, distr_fbb in zip(layers[:n_samples - 1],
                                                             distributions_fbb[:n_samples - 1]))
            if settings.extended_embed:
                params_embeds = [layer.forward_param(distr_fbb, n_subbatch)
                                 for layer, distr_fbb in zip(layers, distributions_fbb)]
                embeds_subbatch = [torch.cat([param, value, obs_embed_subbatch], dim=1)
                                   for param, value in zip(params_embeds, values_embeds)]
            else:
                embeds_subbatch = [torch.cat([value, obs_embed_subbatch], dim=1) for value in values_embeds]

            # List of elements of size n_samples x n_subbatch x embedding_size
            embed_list.append(torch.stack(embeds_subbatch, dim=0).view(n_samples, n_subbatch, -1))
        return indices, embed_list

    def forward(self, batch, previous_sample, previous_hidden):
        obs_embed = self.forward_obs(batch)

        indices, embed_list = self.forward_subbatches(batch, obs_embed, previous_sample)
        indices_sorted, embed_sorted = zip(*sorted(zip(indices, embed_list), reverse=True, key=lambda pair: len(pair[1])))

        embed_padded = util.pad_sequence(embed_sorted, subbatches=True)
        embed_len = []
        for embed in embed_sorted:
            embed_len.extend([len(embed)]*embed.size(1))
        embed_pack = nn.utils.rnn.pack_padded_sequence(embed_padded, embed_len)

        # LSTM
        lstm_output, hidden = self._lstm(embed_pack, previous_hidden)
        if not self.training:
            self._previous_lstm_hidden = hidden
        # Return output and flattened list of indexes
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_output)
        return unpacked, indices_sorted

    def loss(self, batch, optimizer):
        """Computes the loss of the current batch"""
        lstm_output, indices_subbatch = self(batch=batch, previous_sample=None, previous_hidden=None)
        n = 0
        losses = Variable(lstm_output[0].data.new(len(batch)).zero_())
        for indices in indices_subbatch:
            example_trace = batch.traces[indices[0]]
            for step, example_sample in enumerate(example_trace.samples):
                layer = self._get_layer(example_sample)
                proposals_params = layer.proposal(lstm_output[step, n:n+len(indices)])
                # TODO(Lezcano) Implement checking proposal params for NaN
                values_tensor = layer.prior.value.unpack([trace.samples[step].distr_fbb
                                                          for trace in (batch.traces[i] for i in indices)])
                losses[indices] += layer.loss(proposals_params, Variable(values_tensor))
            n += len(indices)
        loss = losses.mean()

        if self.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return float(loss)

    def move_to_cuda(self, device_id=None):
        self.on_cuda = True
        self.cuda_device_id = device_id
        self.cuda(device_id)

    def move_to_cpu(self):
        self.on_cuda = False
        self.cpu()

    def optimize(self, minibatch):
        return self.loss(minibatch, self._optimizer)

    def validation_loss(self, validation_set):
        return self.loss(validation_set, self._optimizer)

    # Inference
    def set_observation(self, observation):
        value = observation.value
        # If we are dealing with sequences, we flatten the value
        if self._obs_sequences:
            value = value.view(-1)
        obs_variable = Variable(value.unsqueeze(0))
        if self._obs_sequences:
            obs_variable = nn.utils.rnn.pack_padded_sequence(obs_variable, [len(value)], batch_first=True)
        self._observation_embedding = self._observation_layer(obs_variable)

    def get_proposal(self, current_sample, previous_sample):
        # New trace
        if previous_sample.distr_fbb is None:
            self._previous_lstm_hidden = None
            previous_sample = None

        current_trace = Batch([Trace([current_sample], None, 0)])
        try:
            lstm_output, _ = self(batch=current_trace,
                                  previous_sample=previous_sample,
                                  previous_hidden=self._previous_lstm_hidden)
        except ValueError:
            return None, None

        address_layer = self._get_layer(current_sample)
        proposal_params = address_layer.proposal(lstm_output[0])
        return address_layer.proposal, proposal_params

    def _get_layer(self, sample):
        try:
            return getattr(self, self._get_layer_name(sample.address))
        except AttributeError:
            if self.training:
                return self._add_layer(sample)
            else:
                raise ValueError("New layer found: {}".format(sample.address))

    def _get_layer_name(self, address):
        return "_address_{}".format(address)

    def _add_layer(self, sample):
        try:
            prior_type = distributionfbb_prior(type(sample.distr_fbb))
        except KeyError:
            error = 'Unsupported distribution: ' + sample.distribution.__class__.__name__
            Logger.logger.log_error(error)
            raise ValueError(error)

        prior = prior_type(sample.distr_fbb, self._embedding_dim)
        if settings.extended_embed:
            type_embedding = self._distribution_embedding.setdefault(prior,
                                                                     Parameter(settings.Tensor(self._embedding_dim).zero_()))
            prior.set_type_embedding(type_embedding)
        layer = Address(prior, sample.distr_fbb, self._out_lstm_dim)
        self.add_module(self._get_layer_name(sample.address), layer)
        Logger.logger.log_info('New layers for address : {}'.format(sample.address))
        if settings.cuda_enabled:
            layer.cuda(settings.cuda_device)
        self._optimizer = torch.optim.Adam(self.parameters())
        return layer

    # Train helper functions
    def _get_observation_layer(self, observation):
        if not hasattr(self, "_observation_layer"):
            if observation.dim() == 3:
                layer = ObserveEmbeddingCNN3D4C(observation.size(), self._embedding_dim)
            elif observation.dim() == 1 and self._obs_embedding_type == "lstm":
                # For now we just support sequences of real numbers
                layer = ObserveEmbeddingLSTM(1, self._embedding_dim)
            elif self._obs_embedding_type == "fc":
                layer = ObserveEmbeddingFC(observation.numel(), self._embedding_dim)
            else:
                error = "Could not set up the Observation Layer. " \
                        "Layer {} cannot be set up with {} dimensions.".format(self._obs_embedding_type, observation.dim())
                Logger.logger.log_error(error)
                raise ValueError(error)

            self.add_module("_observation_layer", layer)
            if settings.cuda_enabled:
                self._observation_layer.cuda(settings.cuda_device)
            self._optimizer = torch.optim.Adam(self.parameters())
        return self._observation_layer

    def _default_value_embed(self, size_subbatch):
        # Value
        return settings.Tensor(size_subbatch, 1, self._embedding_dim).zero_()
