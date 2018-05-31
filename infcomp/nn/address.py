import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from infcomp.distributions.prior_distributions import PriorDistribution
from infcomp.settings import settings
from infcomp.util import weights_init


class Address(nn.Module):
    def __init__(self, prior, distribution_fbb, projection_dim):
        super().__init__()
        # Embedding of the address into R^n with n = embedding_size

        if settings.extended_embed:
            self._addr_embed = Parameter(settings.Tensor(prior.embedding_dim).zero_())

        self.prior = prior
        self.proposal = prior.proposal_type(distribution_fbb=distribution_fbb, input_dim=projection_dim)
        self.apply(weights_init)

    def forward(self):
        # Address is just a wrapper around priors and proposals
        pass

    def forward_param(self, distributions_fbb, n_subbatch):
        """
        Returns a tensor of size n_subbatch x n_elem_embed x width_embed
        where n_eleem_embed are the number of inputs to the RNN at every timestep
        """
        params = [Variable(param) for param in self.prior.unpack(distributions_fbb)]

        address_embed = self._addr_embed.expand(n_subbatch, self.prior.embedding_dim)
        type_embed, param_embed = self.prior(params, n_subbatch)
        return torch.stack([address_embed, type_embed, param_embed], 1)

    def forward_value(self, distributions_fbb):
        value = Variable(self.prior.value.unpack(distributions_fbb))
        return self.prior.value(value)

    def loss(self, params, values):
        return -self.proposal.log_pdf(values, *params).squeeze()

    @staticmethod
    def num_elems_embedding():
        if settings.extended_embed:
            return PriorDistribution.num_elems_embedding() + 1
        else:
            return 0
