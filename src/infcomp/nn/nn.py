import os
import errno

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from infcomp.nn.address import Address
from infcomp.nn.observation import Observation, ObserveEmbeddingFC
from infcomp.data_structures import Trace
from infcomp.settings import settings

from infcomp.parse_flatbuffers import distributionfbb_prior
from infcomp.logger import logger
import infcomp.util as util

from termcolor import colored


class NN(nn.Module):
    def __init__(self, directory="./models", file_name="infcomp-nn"):

        super().__init__()
        # PriorDistr -> EmbeddingVector
        self._distribution_embedding = {}
        # string (sample.address) -> Address
        self._layers = {}

        # Set when the first trace is processed. This will probably change in the future
        self._observation_layer = None
        # Cached embedding for inference. Set in `forward`
        self._observation_embedding = None
        self._previous_embed = None

        self._embedding_dim = settings.embedding_dim
        n = Address.num_elems_embedding()
        no = Observation.num_elems_embedding()
        self._out_lstm_dim = self._embedding_dim

        # 2 addresses (previous + current) + 1 value + the size of the embedding
        lstm_input_dim = (2*n + 1 + no)*self._embedding_dim
        lstm_depth = 2
        self._lstm = nn.LSTM(lstm_input_dim, self._out_lstm_dim, lstm_depth)
        if settings.cuda_enabled:
            self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self.parameters())

        self._dir = directory
        self._file_name = file_name

        # self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.save_nn()

    def save_nn(self):
        print('Updating nn on disk...')
        if not os.path.exists(self._dir):
            print("Creating directory {}".format(self._dir))
            os.mkdir(self._dir)
        torch.save(self, os.path.join(self._dir, self._file_name))

    def forward(self, sub_batch, previous_embed, observations_embed, volatile):
        """Executes one step of the RNN"""
        training = not volatile
        example_trace = sub_batch[0]
        # Prepare the observation to be concat
        observations_embed = observations_embed.unsqueeze(1)
        embed = []
        # Fixme(Lezcano) Is this local variable necessary?
        previous_embed_local = previous_embed
        prev_embeds = []
        for step, example_sample in enumerate(example_trace.samples):
            current_distributions_fbb = [trace.samples[step].distr_fbb for trace in sub_batch]
            layer = self._get_and_add_layer(example_sample, training=training)
            current_params = layer.prior.unpack(current_distributions_fbb)
            current_value = layer.prior.value.unpack(current_distributions_fbb)
            current_embed, current_values_embed = layer.forward(
                [Variable(param, volatile=volatile) for param in current_params],
                Variable(current_value, volatile=volatile))
            current_values_embed = current_values_embed.unsqueeze(1)
            embed.extend(torch.cat([previous_embed_local, observations_embed, current_embed], dim=1))
            previous_embed_local = torch.cat([current_embed, current_values_embed], dim=1)
            prev_embeds.append(previous_embed_local)
        if not training:
            self._previous_embed = previous_embed_local
        n_samples = len(example_trace.samples)
        n_subbatch = len(sub_batch)
        lstm_input = torch.stack(embed, dim=0).view(n_samples, n_subbatch, -1)
        lstm_h, lstm_c = self._default_rnn_input(n_subbatch)
        lstm_output, _ = self._lstm(lstm_input, (Variable(lstm_h, volatile=volatile),
                                                 Variable(lstm_c, volatile=volatile)))
        return lstm_output

    def loss(self, batch, optimizer, volatile):
        """Computes the loss of the current batch"""
        loss = 0.0
        for sub_batch in batch.sub_batches:
            example_trace = sub_batch[0]
            obs_layer = self._get_observation_layer(example_trace.observe)
            # We assume for now that all the observations have the same dimension
            obs_variable = Variable(torch.cat([trace.observe.value for trace in sub_batch]), volatile=volatile)
            obs_embed = obs_layer.forward(obs_variable)

            previous_embed = Variable(self._default_sample_embed(len(sub_batch)), volatile=volatile)
            # Projected Tensor of params
            lstm_output = self.forward(sub_batch, previous_embed, obs_embed, volatile=volatile)
            for step, example_sample in enumerate(example_trace.samples):
                layer = self._layers[example_sample.address]
                proposals_params = layer.proposal.forward(lstm_output[step])
                values_tensor = layer.prior.value.unpack([trace.samples[step].distr_fbb for trace in sub_batch])
                loss += layer.loss(proposals_params, Variable(values_tensor, volatile=volatile))
        loss /= len(batch)

        if not volatile:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.data[0]

    def move_to_cuda(self, device_id=None):
        self.on_cuda = True
        self.cuda_device_id = device_id
        self.cuda(device_id)

    def optimize(self, minibatch):
        return self.loss(minibatch, self._optimizer, volatile=False)

    def validation_loss(self, validation_set):
        return self.loss(validation_set, self._optimizer, volatile=True)

    # Inference
    def set_observation(self, observation):
        obs_variable = Variable(observation.value.unsqueeze(0), volatile=True)
        self._observation_embedding = self._observation_layer(obs_variable)

    def get_proposal(self, current_sample, previous_sample):
        # New trace
        if previous_sample.distr_fbb is None:
            self._previous_embed = Variable(self._default_sample_embed(1), volatile=True)
        current_trace = Trace([current_sample], None)
        lstm_output = self.forward([current_trace], self._previous_embed, self._observation_embedding, volatile=True)
        address_layer = self._layers[current_sample.address]
        proposal_params = address_layer.proposal.forward(lstm_output[0])
        return address_layer.proposal, proposal_params

    def _get_and_add_layer(self, sample, training):
        try:
            return self._layers[sample.address]
        except KeyError:
            if training:
                return self._add_layer(sample)
            else:
                # TODO(Lezcano) Catch this and return a Distribution.None to CPProb
                raise RuntimeError("New layer found: {}".format(sample.address))

    def _add_layer(self, sample):
        try:
            prior_type = distributionfbb_prior(type(sample.distr_fbb))
        except KeyError:
            error = 'Unsupported distribution: ' + sample.distribution.__class__.__name__
            logger.log_error(error)
            raise ValueError(error)

        prior = prior_type(sample.distr_fbb, self._embedding_dim)
        type_embedding = self._distribution_embedding.setdefault(prior,
                                                                 Parameter(settings.Tensor(self._embedding_dim).zero_()))
        prior.set_type_embedding(type_embedding)
        layer = Address(prior, sample.distr_fbb, self._out_lstm_dim)
        self.add_module('address({})'.format(len(self._layers)), layer)
        self._layers[sample.address] = layer
        logger.log(colored('New layers for address : {}'.format(util.truncate_str(sample.address)), 'magenta', attrs=['bold']))
        if settings.cuda_enabled:
            layer.cuda()
        self._optimizer = torch.optim.Adam(self.parameters())
        return layer

    # Train helper functions
    def _get_observation_layer(self, observation):
        if self._observation_layer is None:
            self._observation_layer = ObserveEmbeddingFC(observation.value.numel(), self._embedding_dim)
            if settings.cuda_enabled:
                self._observation_layer.cuda()
        # TODO(Martínez) Check if we need to reconstruct the optimizer because of the observation layer
        self._optimizer = torch.optim.Adam(self.parameters())
        return self._observation_layer

    def _default_sample_embed(self, size_subbatch):
        n = Address.num_elems_embedding()
        # m is the width of the embedding, n is the embedding of:
        # n = address_embed + prior_embed = address + prior_type + prior_params
        # n + 1 = n + value_embed
        return settings.Tensor(size_subbatch, n + 1, self._embedding_dim).zero_()

    def _default_rnn_input(self, size_subbatch):
        lstm_h = settings.Tensor(self._lstm.num_layers, size_subbatch, self._lstm.hidden_size).zero_()
        # We return two copies of the sam eempty value
        return lstm_h, lstm_h
