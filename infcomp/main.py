import argparse
import time
import datetime
import os
from subprocess import PIPE, Popen

from infcomp.train import train
from infcomp.inference import inference
from infcomp.util import file_starting_with
from infcomp.settings import settings
from infcomp.logger import Logger


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
    parser.add_argument('--dir', help='Compile: Directory to save artifacts and logs', default='models', type=str)
    parser.add_argument("--traces_dir", help="Compile: Directory name with the minibatches", type=str)
    parser.add_argument('--cuda', help='Compile / Infer: Use CUDA', action='store_true')
    parser.add_argument('--device', help='Compile / Infer: Selected CUDA device (-1: all, 0: 1st device, 1: 2nd device, etc.)', default=0, type=int)
    parser.add_argument('--save_after_n_traces', help="""Compile: Add a list of numbers and the nn will be saved after
    n traces, for each n in the list. n will be appended to the name of the file.\nExample: --save_after_n_traces 100 1000 10000""", nargs='+', type=int, default=[])
    parser.add_argument("--regularization", help="Compile: Variance-based Regularization.\n Equiv to rho / n in arxiv:1610.02581", default=0, type=float)
    parser.add_argument("--obs_embedding", help="Compile: Observation embedding", choices=["fc", "lstm"], default="fc", type=str)
    parser.add_argument("-s", "--save_file_name", help="Compile: File name to save the NN", type=str)
    parser.add_argument("-l", "--load_file_name", help="Compile / Infer: File name to load the NN", type=str)
    parser.add_argument("-r", "--resume", help="Compile: Resume compilation using the last saved NN (load file name is ignored in that case)", action='store_true')
    parser.add_argument("-p", "--proposals_file_name", help="File name where proposals will be stored", type=str)
    opt = parser.parse_args()

    prefix = "infcomp-nn"
    if opt.cuda is True:
        settings.set_cuda(logger=Logger.logger, device=opt.device)

    if opt.mode == "compile":
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')
        if not os.path.exists(opt.dir):
            print("Creating directory {}".format(opt.dir))
            os.mkdir(opt.dir)
        if opt.tcp_addr is None:
            opt.tcp_addr = "tcp://127.0.0.1:5555"
        if opt.save_file_name is None:
            time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('-%Y%m%d-%H%M%S')
            opt.save_file_name = prefix + time_stamp
        if opt.resume:
            opt.load_file_name = file_starting_with('{0}/{1}'.format(opt.dir, 'infcomp'), -1)

        train(directory=opt.dir,
              save_file_name=opt.save_file_name,
              load_file_name=opt.load_file_name,
              address=opt.tcp_addr,
              regularization=opt.regularization,
              obs_embedding=opt.obs_embedding,
              minibatch_size=opt.minibatch_size,
              save_after_n_traces=opt.save_after_n_traces,
              traces_dir=opt.traces_dir)
    elif opt.mode == "infer":
        if opt.tcp_addr is None:
            opt.tcp_addr = "tcp://0.0.0.0:6666"
        if opt.load_file_name is None:
            opt.load_file_name = file_starting_with(os.path.join(opt.dir, prefix), -1)
            print(opt.load_file_name)

        inference(nn_file_name=opt.load_file_name,
                  address=opt.tcp_addr,
                  proposals_file_name=opt.proposals_file_name)


if __name__ == "__main__":
    main()
