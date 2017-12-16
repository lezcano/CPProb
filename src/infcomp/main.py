import argparse
import time
import datetime
import os

from infcomp.train import train
from infcomp.inference import inference
from infcomp.util import file_starting_with
from infcomp.settings import settings
from infcomp.logger import logger



def main():
    parser = argparse.ArgumentParser(description="Inference Compilation NN")
    parser.add_argument("-m", "--mode",
                        help="Compile: Compile a NN to use in inference mode.\n"
                             "Infer: Perform inference compilation.\n",
                        choices=["compile", "infer"],
                        required=True,
                        type=str)
    parser.add_argument("-a", "--tcp_addr",
                        help= "Address and port to connect with the NN.\n"
                              "Defaults:\n"
                              "  Compile: tcp://127.0.0.1:5555\n"
                              "  Infer:   tcp://0.0.0.0:6666",
                        type=str)
    parser.add_argument("--minibatch_size", help="Compile: Minibatch size.", default=64, type=int)
    parser.add_argument('--dir', help='Directory to save artifacts and logs', default='models')
    parser.add_argument('--cuda', help='use CUDA', action='store_true') # By default it is false
    parser.add_argument('--device', help='selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=0, type=int)
    parser.add_argument("-s", "--save_file_name", help="File name to save the NN", type=str)
    parser.add_argument('-w', action='store_true')
    parser.add_argument("-l", "--load_file_name", help="File name to load the NN", type=str)
    parser.add_argument("-r", help="Resume compilation using the last saved NN (load file name is ignored in that case)", action='store_true')
    parser.add_argument("--validation_size", help="Compile: Validation set size.", default=128, type=int)
    parser.add_argument("--n_batches",
                        help="Compile: Number of minibatches to process. Pass 0 to train until killed.",
                        default=0,
                        type=int)
    opt = parser.parse_args()

    if opt.cuda is True:
        settings.set_cuda(logger=logger, device=opt.device)

    if opt.mode == "compile":
        if opt.tcp_addr is None:
            opt.tcp_addr = "tcp://127.0.0.1:5555"
        if opt.save_file_name is None:
            time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')
            opt.save_file_name = 'infcomp-nn' + time_stamp
        if opt.r is True:
            opt.load_file_name = file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp'), -1)
            pass

        train(directory=opt.dir,
              save_file_name=opt.save_file_name,
              load_file_name=opt.load_file_name,
              address=opt.tcp_addr,
              minibatch_size=opt.minibatch_size,
              validation_size=opt.validation_size,
              n_batches=opt.n_batches)
    elif opt.mode == "infer":
        if opt.tcp_addr is None:
            opt.tcp_addr = "tcp://0.0.0.0:6666"
        if opt.load_file_name is None:
            opt.file_name = file_starting_with(os.path.join(opt.dir, 'infcomp-nn'), -1)

        inference(file_name=opt.file_name,
                  address=opt.tcp_addr)


if __name__ == "__main__":
    main()
