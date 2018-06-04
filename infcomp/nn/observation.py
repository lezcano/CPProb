import torch
import torch.nn as nn

from infcomp.settings import settings
from infcomp.util import weights_init


class Observation(nn.Module):
    def __init__(self):
       super().__init__()

    def forward(self, x):
        raise NotImplementedError


class ObserveEmbeddingFC(Observation):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._input_dim = input_dim
        self._module = nn.Sequential(
            nn.Linear(input_dim, output_dim//8),
            nn.ReLU(),
            nn.Linear(output_dim//8, output_dim//4),
            nn.ReLU(),
            nn.Linear(output_dim//4, output_dim//2),
            nn.ReLU(),
            nn.Linear(output_dim//2, output_dim),
            nn.ReLU()
        )
        self.apply(weights_init)

    def forward(self, x):
        # Dimensions: BatchxInputDim
        if x.dim() > 2:
            x = x.view(-1, self._input_dim)
        return self._module(x)


class ObserveEmbeddingLSTM(Observation):
    def __init__(self, input_dim, output_dim, depth=2):
        super().__init__()
        self._lstm = nn.LSTM(input_dim, output_dim, depth, batch_first=True)
        self.apply(weights_init)

    def forward(self, x):
        lstm_output, _ = self._lstm(x, None)
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_output)

        # Return just the last output of the LSTM
        out = unpacked.new_empty(unpacked.size()[1:])
        for i, l in enumerate(unpacked_len):
            out[i] = unpacked[l.item() - 1, i]
        return out


class ObserveEmbeddingCNN3D4C(Observation):
    def __init__(self, input_size, output_dim):
        super().__init__()
        if len(input_size) != 3:
            raise RuntimeError("Input size for 3D4C convolution is {} instead of 3.".format(len(input_size)))
        self._cnn = nn.Sequential(
            nn.Conv3d(1, 4, 3),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )

        # Compute CNN output dim:
        cnn_output = self._cnn(torch.randn(1, 1, *input_size, device=settings.device))
        self.cnn_output_dim = cnn_output.numel()

        self._projection = nn.Sequential(
            nn.Linear(self.cnn_output_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU()
        )
        self.apply(weights_init)

    def forward(self, x):
        # This indicates that there are no channel dimensions and we have BxDxHxW
        # Add a channel dimension of 1 after the batch dimension so that we have BxCxDxHxW
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self._cnn(x)
        x = x.view(-1, self.cnn_output_dim)
        return self._projection(x)
