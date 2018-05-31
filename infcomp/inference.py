import torch
from infcomp.server import Server
from infcomp.logger import Logger
from infcomp.util import load_nn
from infcomp.actions import FinishInference


def _run(address, nn, proposals_file):
    with Server(address) as sv:
        while True:
            action = sv.get_action()
            action.run(nn, sv, proposals_file)
            if isinstance(action, FinishInference):
                break


def inference(nn_file_name, address, proposals_file_name):
    # If proposals_file_name is None the proposals will not be stored
    with torch.no_grad():
        nn = load_nn(nn_file_name)
        nn.eval()
        Logger.logger.log_infer_begin(address)

        if proposals_file_name is not None:
            with open(proposals_file_name, 'w+') as proposals_file:
                _run(address, nn, proposals_file)
        else:
            _run(address, nn, None)


