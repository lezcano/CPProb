import torch.nn as nn
from infcomp.util import weights_init


from infcomp.distributions.embeddings import (RealEmbedding,
                                              RealInIntervalEmbedding,
                                              bounded_integer_builder)

from infcomp.distributions.proposal_distributions import (ProposalNormal,
                                                          ProposalTruncated,
                                                          ProposalMixture,
                                                          ProposalDiscrete,
                                                          truncated_builder,
                                                          mixture_builder)


class PriorDistribution(nn.Module):
    def __init__(self, value_type, embedding_dim, proposal):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._value = value_type(embedding_dim=embedding_dim)
        self.proposal = proposal
        self.apply(weights_init)

    def forward(self, x):
        return self._value(x)

    def unpack(self, distributions_fbb):
        return self._value.unpack(distributions_fbb)


class PriorNormal(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim, projection_dim, proposal_type=ProposalNormal):
        proposal = proposal_type(distribution_fbb, projection_dim)
        super().__init__(value_type=RealEmbedding, embedding_dim=embedding_dim, proposal=proposal)


class PriorDiscrete(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim, projection_dim, proposal_type=ProposalDiscrete):
        proposal = proposal_type(distribution_fbb, projection_dim)
        min_val = distribution_fbb.Min()
        max_val = distribution_fbb.Max()
        super().__init__(value_type=bounded_integer_builder(min_val, max_val),
                         embedding_dim=embedding_dim,
                         proposal=proposal)


class PriorUniformDiscrete(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim, projection_dim, proposal_type=ProposalDiscrete):
        proposal = proposal_type(distribution_fbb, projection_dim)
        min_val = distribution_fbb.Min()
        max_val = distribution_fbb.Max()
        super().__init__(value_type=bounded_integer_builder(min_val, max_val),
                         embedding_dim=embedding_dim,
                         proposal=proposal)


class PriorUniformContinuous(PriorDistribution):
    def __init__(self, distribution_fbb, embedding_dim, projection_dim,
                 proposal_type=truncated_builder(mixture_builder(ProposalNormal, n=8))):
        proposal = proposal_type(distribution_fbb, projection_dim)
        super().__init__(value_type=RealInIntervalEmbedding,
                         embedding_dim=embedding_dim,
                         proposal=proposal)
