"""
A NN has way too many parameters to fine-tune them on an individual basis.
Doing so would require passing too many parameters to every function / object
 on the object constructor.
This file has the configuration of the NN.
This configuration is set at the start of the program once and then its kept constant
 throughout the whole execution of the program.
"""
import torch


class Settings:

    def __init__(self):
        self._cuda_enabled = False
        self._cuda_device = None
        self._embedding_dim = 128
        self._dropout = 0
        self.Tensor = torch.FloatTensor
        self._extended_embed = False

    @property
    def cuda_enabled(self):
        return self._cuda_enabled

    @property
    def cuda_device(self):
        return self._cuda_device

    @property
    def extended_embed(self):
        return self._extended_embed

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def dropout(self):
        return self._dropout

    def set_cuda(self, logger, cuda=True, device=0):
        if cuda:
            if torch.cuda.is_available():
                self._cuda_enabled = True
                self._cuda_device = device
                torch.cuda.set_device(device)
                torch.backends.cudnn.enabled = True
                #torch.backends.cudnn.fastest = True
                self.Tensor = torch.cuda.FloatTensor
            else:
                logger.log_warning("Cuda is not available, running on CPU.")
        else:
            self._cuda_enabled = False
            self.Tensor = torch.FloatTensor


settings = Settings()
