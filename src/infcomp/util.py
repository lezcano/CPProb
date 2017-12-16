from glob import glob
from infcomp.logger import logger
import sys
from infcomp.settings import settings
import torch


def truncate_str(s, length=80):
    return (s[:length] + '...') if len(s) > length else s


def file_starting_with(pattern, n):
    try:
        return sorted(glob(pattern + '*'))[n]
    except:
        logger.log_error('Cannot find file')
        sys.exit(1)


def load_nn(file_name):
    try:
        if settings.cuda_enabled:
            nn = torch.load(file_name)
            if settings.cuda_device == -1:
                settings.cuda_device = torch.cuda.current_device()

            nn.move_to_cuda(settings.device_id)
        else:
            nn = torch.load(file_name, map_location=lambda storage, loc: storage)
            if nn.on_cuda:
                logger.log_warning('Loading CUDA artifact to CPU')
                logger.log()
                nn.move_to_cpu()
        return nn
    except Exception as e:
        logger.log_error('Cannot load file')
        raise

