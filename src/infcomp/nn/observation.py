import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from infcomp.settings import settings


class Observation(nn.Module):
    def __init__(self):
       super().__init__()

    @staticmethod
    def num_elems_embedding():
        return 1

    def forward(self, x):
        raise NotImplemented()


class ObserveEmbeddingFC(Observation):
    def __init__(self, input_width, embedding_dim):
        super().__init__()
        self.input_width = input_width
        self.lin1 = nn.Linear(self.input_width, embedding_dim//2)
        self.lin2 = nn.Linear(embedding_dim//2, embedding_dim)
        self.drop = nn.Dropout(settings.dropout)
        init.xavier_uniform(self.lin1.weight, gain=np.sqrt(2.0))
        init.xavier_uniform(self.lin2.weight, gain=np.sqrt(2.0))

    def forward(self, x):
        x = F.relu(self.lin1(x.view(-1, self.input_width)))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        return x
