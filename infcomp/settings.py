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
        self._embedding_dim = 128
        self._cuda_enabled = False
        self._device = torch.device('cpu')

    @property
    def cuda_enabled(self):
        return self._cuda_enabled

    @property
    def device(self):
        return self._device

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def tensor(self, data=[], dtype=torch.float, device=None, requires_grad=False):
        return torch.tensor(data, dtype=dtype, device=self.device, requires_grad=requires_grad)

    def set_cuda(self, logger, cuda=True, device_index=0):
        if cuda:
            if torch.cuda.is_available():
                self._cuda_enabled = True
                self._device = torch.device(type='cuda', index=device_index)
                torch.backends.cudnn.enabled = True
                #torch.backends.cudnn.fastest = True
                return
            else:
                logger.log_warning("Cuda is not available, running on CPU.")
        self._cuda_enabled = False
        self._device = torch.device('cpu')
        torch.backends.cudnn.enabled = False


settings = Settings()
