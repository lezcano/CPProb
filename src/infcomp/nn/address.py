import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from infcomp.distributions.prior_distributions import PriorDistribution
from infcomp.settings import settings


class Address(nn.Module):
    def __init__(self, prior, distribution_fbb, projection_dim):
        super().__init__()
        # Embedding of the address into R^n with n = embedding_size
        self._addr_embed = Parameter(settings.Tensor(prior.embedding_dim).zero_())
        self.prior = prior
        self.proposal = prior.proposal_type(distribution_fbb=distribution_fbb, input_dim=projection_dim)

    def forward(self, params, value):
        """
        Returns a tensor of size n_subbatch x n_elem_embed x width_embed
        where n_eleem_embed are the number of inputs to the RNN at every timestep
        """

        n_subbatch = value.size()[0]
        address_embed = self._addr_embed.expand(n_subbatch, self.prior.embedding_dim)
        type_embed, param_embed = self.prior.forward(params, n_subbatch)
        value_embed = self.prior.value.forward(value)
        return torch.stack([address_embed, type_embed, param_embed], 1), value_embed

    def loss(self, params, values):
        return -torch.sum(self.proposal.log_pdf(values, *params))

    @staticmethod
    def num_elems_embedding():
        return PriorDistribution.num_elems_embedding() + 1
