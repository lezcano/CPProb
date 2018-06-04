import torch
import torch.nn as nn
from infcomp.settings import settings


def unpack_scalar(distributions_fbb):
    return settings.tensor([fb.Value() for fb in distributions_fbb]).unsqueeze(1)


def unpack_vector(distributions_fbb):
    return torch.stack([settings.tensor([fb.Value(i) for i in range(fb.ValueLength())])
                        for fb in distributions_fbb])


class EmbeddingType(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

    def forward(self, value):
        raise NotImplementedError

    def unpack(self, distributions_fbb):
        raise NotImplementedError


class DefaultEmbeddingType(EmbeddingType):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(embedding_dim)
        self._unpack = unpack_scalar if input_dim == 1 else unpack_vector
        # Specialise in the child classes
        self._model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim//8),
            nn.ReLU(),
            nn.Linear(embedding_dim//8, embedding_dim//4),
            nn.ReLU(),
            nn.Linear(embedding_dim//4, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, embedding_dim),
            nn.ReLU()
        )

    def forward(self, value):
        return self._model(value)

    def unpack(self, distributions_fbb):
        return self._unpack(distributions_fbb)


class RealEmbedding(DefaultEmbeddingType):
    def __init__(self, embedding_dim):
        super().__init__(input_dim=1, embedding_dim=embedding_dim)


class RealPositiveEmbedding(DefaultEmbeddingType):
    def __init__(self, embedding_dim):
        super().__init__(input_dim=1, embedding_dim=embedding_dim)


class RealInIntervalEmbedding(DefaultEmbeddingType):
    def __init__(self, embedding_dim):
        super().__init__(input_dim=1, embedding_dim=embedding_dim)


class PointInSimplexEmbedding(DefaultEmbeddingType):
    def __init__(self, input_dim, embedding_dim):
        super().__init__(input_dim=input_dim, embedding_dim=embedding_dim)


class IntegerEmbedding(DefaultEmbeddingType):
    def __init__(self, embedding_dim):
        super().__init__(input_dim=1, embedding_dim=embedding_dim)


class BoundedIntegerEmbedding(EmbeddingType):
    def __init__(self, min_val, max_val, embedding_dim):
        super().__init__(embedding_dim=embedding_dim)
        self._min = min_val
        self._max = max_val
        size = max_val - min_val + 1
        self._embed = torch.nn.Embedding(size, embedding_dim)

    def unpack(self, distributions_fbb):
        return settings.tensor([fb.Value() for fb in distributions_fbb], dtype=torch.long)

    def forward(self, value):
        return self._embed(value - self._min)


def bounded_integer_builder(min_val, max_val):
    def wrapper(embedding_dim):
        return BoundedIntegerEmbedding(min_val=min_val,
                                       max_val=max_val,
                                       embedding_dim=embedding_dim)
    return wrapper
