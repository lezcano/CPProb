from infcomp.client import RequesterClient, RequesterFile

from infcomp.nn.nn import NN
from infcomp.logger import Logger
from infcomp.util import load_nn
from infcomp.settings import settings
from infcomp.util import save_if_its_time


def train(directory, save_file_name, load_file_name, address, regularization,
          obs_embedding, minibatch_size, save_after_n_traces, traces_dir=None):
    if load_file_name is None:
        nn = NN(directory=directory, file_name=save_file_name, regularization=regularization, obs_embedding_type=obs_embedding)
        Logger.logger.log_info("New nn will be saved to: {}/{}".format(directory, save_file_name))
    else:
        nn = load_nn(load_file_name)
        Logger.set(nn.logger)
        Logger.logger.log_info("Resuming previous artifact: {}/{}".format(directory, load_file_name))
    if settings.cuda_enabled:
        nn.move_to_cuda(settings.cuda_device)
    nn.train()
    save_after_n_traces.sort(reverse=True)

    if not traces_dir:
        requester_class = RequesterClient
        params = [address]
    else:
        requester_class = RequesterFile
        params = [traces_dir]
    with requester_class(*params) as requester:
        errors = True
        try:
            i = 0
            n_processed_traces = 0
            best_loss_freq = 10

            Logger.logger.log_training_begin(*params)
            while not save_if_its_time(nn, save_after_n_traces, n_processed_traces):
                minibatch = requester.minibatch(minibatch_size)
                if settings.cuda_enabled:
                    minibatch.cuda(settings.cuda_device)

                train_loss = nn.optimize(minibatch)
                Logger.logger.log_training(len(minibatch), train_loss, nn)
                if (i + 1) % best_loss_freq == 0:
                    Logger.logger.log_training_best()

                i += 1
                n_processed_traces += len(minibatch)
            errors = False
        except KeyboardInterrupt:
            pass
        finally:
            # Try to save if there was an exception
            if errors:
                nn.__exit__(None, None, None)
