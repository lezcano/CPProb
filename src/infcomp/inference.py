import torch

from infcomp.server import Server
from infcomp.logger import logger
from infcomp.util import load_nn

def inference(file_name, address):
    logger.log_config()
    nn = load_nn(file_name)
    nn.eval()
    with Server(address) as sv:
        while True:
            action = sv.get_action()
            action.run(nn, sv)
