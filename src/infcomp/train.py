from infcomp.client import Client

from infcomp.nn.nn import NN
from infcomp.logger import logger, Logger
from termcolor import colored
from infcomp.util import load_nn
from infcomp.settings import settings
import torch


def train(directory, save_file_name, load_file_name, address, minibatch_size, validation_size, n_batches):
    if load_file_name is None:
        nn = NN(directory=directory, file_name=save_file_name)
        if settings.cuda_enabled:
            nn.move_to_cuda(settings.cuda_device)
    else:
        nn = load_nn(load_file_name)
        logger.log(colored('Resuming previous artifact:  {}/{}'.format(directory, save_file_name), 'blue', attrs=['bold']))
        #Logger.set(nn.logger) TODO (Martínez) Check how the logger can be saved into file (requires managing the state of the log file)
    nn.train()

    with Client(address) as request:
        try:
            logger.log(colored('New nn will be saved to: {}/{}'.format(directory, save_file_name), 'blue', attrs=['bold']))

            # Ask for Validation set
            validation_set = request.validation_set(validation_size)
            if settings.cuda_enabled:
                validation_set.cuda()
            i = 0
            valid_loss_freq = 5

            logger.log_compile_begin(address)

            while True:
                minibatch = request.minibatch(minibatch_size)
                if settings.cuda_enabled:
                    minibatch.cuda(settings.cuda_device)

                train_loss = nn.optimize(minibatch)
                logger.log_statistics(len(minibatch), train_loss)
                if (i+1) % valid_loss_freq == 0:
                    validation_loss = nn.validation_loss(validation_set)
                    logger.log_statistics_validation(validation_loss)

                i = i + 1
                if n_batches != 0 and i == n_batches:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            nn.__exit__(None, None, None)

