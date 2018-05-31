import torch
import torch.nn as nn
import torch.nn.init as init
from infcomp.settings import settings
from torch.autograd import Variable
from torch import LongTensor


def unpack_scalar(distributions_fbb, name):
    return settings.Tensor([getattr(fb, name)() for fb in distributions_fbb]).unsqueeze(1)


def unpack_vector(distributions_fbb, name):
    return torch.stack([settings.Tensor([getattr(fb, name)(i)
                                for i in range(getattr(fb, "{}Length".format(name))())])
                        for fb in distributions_fbb])


class EmbeddingType(nn.Module):
    def __init__(self, name, distribution_fbb=None):
        super().__init__()

    def init_module(self, embedding_dim):
        raise NotImplementedError

    def forward(self, value):
        raise NotImplementedError

    def unpack(self, distributions_fbb):
        raise NotImplementedError


class DefaultEmbeddingType(EmbeddingType):
    def __init__(self, name, input_dim, distribution_fbb=None):
        super().__init__(name, distribution_fbb)
        self._name = name
        self._input_dim = input_dim
        self.value = self.unpack([distribution_fbb]).squeeze() if distribution_fbb is not None else None

    def init_module(self, embedding_dim):
        # Specialise in the child classes
        self.add_module("_model", nn.Sequential(
            nn.Linear(self._input_dim, embedding_dim//8),
            #nn.BatchNorm1d(embedding_dim//8),
            nn.ReLU(),
            nn.Linear(embedding_dim//8, embedding_dim//4),
            #nn.BatchNorm1d(embedding_dim//4),
            nn.ReLU(),
            nn.Linear(embedding_dim//4, embedding_dim//2),
            #nn.BatchNorm1d(embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, embedding_dim),
            #nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        ))

    def forward(self, value):
        if value is None:
            value = Variable(self.value)
        return self._model(value)

    def unpack(self, distributions_fbb):
        if self._input_dim == 1:
            return unpack_scalar(distributions_fbb, self._name)
        else:
            return unpack_vector(distributions_fbb, self._name)


class RealDefaultEmbedding(DefaultEmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class RealPositiveDefaultEmbedding(DefaultEmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class RealInIntervalDefaultEmbedding(DefaultEmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class PointInSimplexDefaultEmbedding(DefaultEmbeddingType):
    def __init__(self, name, input_dim, distribution_fbb=None):
        super().__init__(name=name, input_dim=input_dim, distribution_fbb=distribution_fbb)


class IntegerDefaultEmbedding(DefaultEmbeddingType):
    def __init__(self, name, distribution_fbb=None):
        super().__init__(name=name, input_dim=1, distribution_fbb=distribution_fbb)


class BoundedIntegerEmbedding(EmbeddingType):
    def __init__(self, name, min_val, max_val, distribution_fbb=None):
        super().__init__(name, distribution_fbb)
        self._name = name
        self._min = min_val
        self._max = max_val
        self.value = self.unpack([distribution_fbb]).squeeze() if distribution_fbb is not None else None

    def unpack(self, distributions_fbb):
        ret = LongTensor([getattr(fb, self._name)() for fb in distributions_fbb])
        if settings.cuda_enabled:
            return ret.cuda(settings.cuda_device)
        else:
            return ret

    def init_module(self, embedding_dim):
        size = self._max - self._min + 1
        self.add_module("_embed", torch.nn.Embedding(size, embedding_dim))

    def forward(self, value):
        if value is None:
            value = Variable(self.value)
        return self._embed(value - self._min)


def bounded_integer_builder(min_val, max_val):
    def wrapper(name, distribution_fbb):
        return BoundedIntegerEmbedding(name=name,
                                       min_val=min_val,
                                       max_val=max_val,
                                       distribution_fbb=distribution_fbb)
    return wrapper


#class ListEmbedding(nn.ModuleList):
#    def __init__(self, name, embeddings):
#        # No distribution_fbb for now, we do not allow it as a hyperparameter for now
#        super().__init__(embeddings)
#        self._name = name
#
#    def init_module(self, embedding_dim):
#        for embedding in self:
#            embedding.init_module(embedding_dim)
