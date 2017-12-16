import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools


from infcomp.distributions.embeddings import (RealEmbedding,
                                              RealPositiveEmbedding,
                                              RealInIntervalEmbedding,
                                              PointInSimplexEmbedding,
                                              IntegerEmbedding)

from infcomp.settings import settings
from infcomp.distributions.proposal_distributions import (ProposalNormal,
                                                          ProposalDiscrete,
                                                          ProposalMixture,
                                                          ProposalTruncated)


class PriorDistribution(nn.Module):
    def __init__(self, params, hyperparams, value_type, embedding_dim, proposal_type):
        super().__init__()
        # Embedding vector for the NN
        self._type_embedding = None

        self.embedding_dim = embedding_dim
        self._embedding_dim_params = embedding_dim // (len(params) + len(hyperparams))
        self._params = nn.ModuleList(params)
        self._hyperparams = nn.ModuleList(hyperparams)
        for param in itertools.chain(self._hyperparams, self._params):
            param.init_module(self._embedding_dim_params)
        self.value = value_type(name="Value")
        self.value.init_module(self.embedding_dim)
        self.proposal_type = proposal_type

    def set_type_embedding(self, type_embedding):
        self._type_embedding = type_embedding

    def forward(self, params, n_subbatch):
        all_params_embed = self._cat_params([hyper.forward() for hyper in self._hyperparams],
                                            [param_module.forward(tensor_param)
                                             for param_module, tensor_param in zip(self._params, params)], n_subbatch)

        type_embed_batch = self._type_embedding.expand(n_subbatch, self.embedding_dim)
        return type_embed_batch, all_params_embed

    def _cat_params(self, hyperparam_embeds, params_embeds, n_subbatch):
        hyperparams_expanded = [hyperparam.expand(n_subbatch, self._embedding_dim_params)
                                for hyperparam in hyperparam_embeds]
        ret = torch.cat(hyperparams_expanded + params_embeds, 1)
        pad = self.embedding_dim % ret.size(1)
        if pad != 0:
            ret = torch.cat((ret, Variable(torch.zeros(n_subbatch, pad))), dim=1)
        return ret

    def unpack(self, distributions_fbb):
        return [param.unpack(distributions_fbb) for param in self._params]

    @staticmethod
    def num_elems_embedding():
        return 2


class PriorNormal(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim=settings.embedding_dim, proposal_type=ProposalNormal):
        params = [
            RealEmbedding(name="Mean"),
            RealPositiveEmbedding(name="Std")
        ]

        super().__init__(params=params, hyperparams=[], value_type=RealEmbedding, embedding_dim=embedding_dim, proposal_type=proposal_type)


class PriorUniformDiscrete(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim=settings.embedding_dim, proposal_type=ProposalDiscrete):
        hyperparams = [
            RealEmbedding(name="Min", distribution_fbb=distribution_fbb),
            RealEmbedding(name="Max", distribution_fbb=distribution_fbb)
        ]
        super().__init__(params=[], hyperparams=hyperparams, value_type=IntegerEmbedding, embedding_dim=embedding_dim, proposal_type=proposal_type)


class PriorDiscrete(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim=settings.embedding_dim, proposal_type=ProposalDiscrete):
        params = [
            PointInSimplexEmbedding(name="Probabilities", input_dim=distribution_fbb.Max() - distribution_fbb.Min() + 1)
        ]

        hyperparams = [
            RealEmbedding(name="Min", distribution_fbb=distribution_fbb),
            RealEmbedding(name="Max", distribution_fbb=distribution_fbb)
        ]
        super().__init__(params=params,
                         hyperparams=hyperparams,
                         value_type=IntegerEmbedding,
                         embedding_dim=embedding_dim,
                         proposal_type=proposal_type)


def mixture_builder(distribution_fbb, input_dim):
    return ProposalMixture(distribution_fbb=distribution_fbb,
                           input_dim=input_dim,
                           distribution_type=ProposalNormal,
                           n=10)


def truncated_mixture_builder(distribution_fbb, input_dim):
    return ProposalTruncated(distribution_fbb=distribution_fbb,
                             input_dim=input_dim,
                             distribution_type=mixture_builder)


class PriorUniformContinuous(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim, proposal_type=truncated_mixture_builder):
        hyperparams = [
            RealEmbedding(name="Min", distribution_fbb=distribution_fbb),
            RealEmbedding(name="Max", distribution_fbb=distribution_fbb)
        ]
        super().__init__(params=[],
                         hyperparams=hyperparams,
                         value_type=RealInIntervalEmbedding,
                         embedding_dim=embedding_dim,
                         proposal_type=proposal_type)
