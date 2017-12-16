import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from infcomp.settings import settings
from torch.autograd import Variable


def unpack_scalar(distributions_fbb, name):
    return settings.Tensor([getattr(fb, name)() for fb in distributions_fbb]).unsqueeze(1)


def unpack_vector(distributions_fbb, name):
    return torch.stack([settings.Tensor([getattr(fb, name)(i)
                                for i in range(getattr(fb, "{}Length".format(name))())])
                        for fb in distributions_fbb])


class EmbeddingType(nn.Module):
    def __init__(self, name, input_dim, distribution_fbb):
        super().__init__()
        self._name = name
        self._input_dim = input_dim
        self.value = self.unpack([distribution_fbb]).squeeze() if distribution_fbb is not None else None
        self.lin1 = None

    def init_module(self, embedding_dim):
        # Specialise in the child classes
        self.lin1 = nn.Linear(self._input_dim, embedding_dim)
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
        self.add_module("lin1_module", self.lin1)

    def forward(self, value=None):
        if value is None:
            value = Variable(self.value)
        return F.relu(self.lin1(value))

    def unpack(self, distributions_fbb):
        if self._input_dim == 1:
            return unpack_scalar(distributions_fbb, self._name)
        else:
            return unpack_vector(distributions_fbb, self._name)


# TODO: Right now the embedding layer for every type is the same, change
class RealEmbedding(EmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class RealPositiveEmbedding(EmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class RealInIntervalEmbedding(EmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class PointInSimplexEmbedding(EmbeddingType):
    def __init__(self, name, input_dim, distribution_fbb=None):
        super().__init__(name=name, input_dim=input_dim, distribution_fbb=distribution_fbb)


class IntegerEmbedding(EmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)
