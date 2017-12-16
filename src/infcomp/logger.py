import logging
import re
import cpuinfo
import time
import torch
from termcolor import colored
from infcomp.settings import settings


def format_seconds(seconds):
    return time.strftime("%dd %Hh %Mm %Ss", time.gmtime(seconds))


def format_trace_num(trace_num):
    # TODO(Lezcano) Simplify this format.format
    return '{:5}'.format('{:,}'.format(trace_num))


def format_tps(traces_per_sec):
    # TODO(Lezcano) Simplify this format.format
    return '{:3}'.format('{:,}'.format(int(traces_per_sec)))

def remove_non_ascii(s):
    remove_non_ascii.regex = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    s = remove_non_ascii.regex.sub('', s)
    return ''.join(i for i in s if ord(i) < 128)


class Logger:
    def __init__(self, file_name='nn.log'):
        self._file_name = file_name

        self._logger = logging.getLogger()
        if file_name is not None:
            logger_file_handler = logging.FileHandler(file_name)
            logger_file_handler.setFormatter(logging.Formatter('%(asctime)s %(funcName)s: %(message)s'))
            self._logger.addHandler(logger_file_handler)
        self._logger.setLevel(logging.INFO)

        # TODO(Lezcano) Make dynamic
        self.time_row_width = 15
        self.trace_row_width = 7

        self.time_init = time.time()
        self.total_traces = 0

        self.last_time = time.time()
        self.time_last_batch = self.last_time

        self.last_validation_loss = float("inf")
        self.valid_loss_str = ""

        self.train_loss_best = float("inf")
        self.valid_loss_best = float("inf")

        self.time_best = 0
        self.trace_best = 0

    @property
    def total_time(self):
        return time.time() - self.time_init

    @property
    def train_loss_best_str(self):
        return colored('{:+.6e}  '.format(self.train_loss_best), 'green', attrs=['bold'])

    @property
    def valid_loss_best_str(self):
        return colored('{:+.6e}  '.format(self.valid_loss_best), 'green', attrs=['bold'])

    def log(self, line=''):
        print(line)
        self._logger.info(remove_non_ascii(line))

    def log_error(self, line=''):
        line = colored('Error: ' + line, 'red', attrs=['bold'])
        print(line)
        self._logger.error(remove_non_ascii(line))

    def log_warning(self, line=''):
        line = colored('Warning: ' + line, 'red', attrs=['bold'])
        print(line)
        self._logger.warning(remove_non_ascii(line))

    def log_config(self):
        line0 = self.get_config()
        self._logger.info('')
        self._logger.info(remove_non_ascii(line0))
        self._logger.info('')

    def log_infer_begin(self):
        line1 = 'TPS      │ Max. TPS │ Total traces'
        line2 = '─────────┼──────────┼─────────────'
        self._logger.info('')
        self._logger.info(remove_non_ascii(line1))
        self._logger.info(remove_non_ascii(line2))

    def log_infer(self, traces_per_sec_str, max_traces_per_sec_str, total_traces):
        line = '{0} │ {1} │ {2}      \n'.format(traces_per_sec_str, max_traces_per_sec_str, total_traces)
        #sys.stdout.write(line)
        #sys.stdout.flush()
        print(line)
        self._logger.info(remove_non_ascii(line))

    def log_compile_begin(self, server):
        line = colored('Training from ' + server, 'blue', attrs=['bold'])
        print(line)
        self._logger.info(remove_non_ascii(line))
        self._print_header()

    def log_config(self):
        ret = []
        ret.append('PyTorch {}'.format(torch.__version__))
        cpu_info = cpuinfo.get_cpu_info()
        if 'brand' in cpu_info:
            ret.append('CPU           : {}'.format(cpu_info['brand']))
        else:
            ret.append('CPU           : unknown')
        if 'count' in cpu_info:
            ret.append('CPU count     : {0} (logical)'.format(cpu_info['count']))
        else:
            ret.append('CPU count     : unknown')
        if torch.cuda.is_available():
            ret.append('CUDA          : available')
            ret.append('CUDA devices  : {0}'.format(torch.cuda.device_count()))
            """
            if settings.cuda_enabled:
                if cuda_device == -1:
                    ret.append('CUDA selected : all')
                else:
                    ret.append('CUDA selected : {0}'.format(cuda_device))
            """
        else:
            ret.append('CUDA          : not available')
        if settings.cuda_enabled:
            ret.append('Running on    : CUDA')
        else:
            ret.append('Running on    : CPU')
        return '\n'.join(ret)

    @staticmethod
    def set(new_logger):
        global logger
        logger = new_logger

    def log_statistics(self, batch_length, train_loss):
        now = time.time()
        time_spent_last_batch = now - self.time_last_batch
        traces_per_sec = batch_length / time_spent_last_batch
        self.time_last_batch = now

        self.total_traces += batch_length

        if train_loss < self.train_loss_best:
            self.train_loss_best = train_loss
            train_loss_str = colored('{:+.6e} ▼'.format(train_loss), 'green', attrs=['bold'])
            self.time_best = self.total_time
            self.trace_best = self.total_traces
        elif train_loss < self.last_validation_loss:
            train_loss_str = colored('{:+.6e}  '.format(train_loss), 'green')
        elif train_loss > self.last_validation_loss:
            train_loss_str = colored('{:+.6e}  '.format(train_loss), 'red')
        else:
            train_loss_str = '{:+.6e}  '.format(train_loss)

        self._log_row_data("current",
                           self.total_time,
                           self.total_traces,
                           train_loss_str,
                           self.valid_loss_str,
                           traces_per_sec)

    def log_statistics_validation(self, valid_loss):
        self.time_last_batch = time.time()

        if valid_loss < self.valid_loss_best:
            self.valid_loss_best = valid_loss
            self.valid_loss_str = colored('{:+.6e} ▼'.format(valid_loss), 'green', attrs=['bold'])
        elif valid_loss < self.last_validation_loss:
            self.valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'green')
        elif valid_loss > self.last_validation_loss:
            self.valid_loss_str = colored('{:+.6e}  '.format(valid_loss), 'red')
        else:
            self.valid_loss_str = '{:+.6e}  '.format(valid_loss)

        self.last_validation_loss = valid_loss

        self._print_mid_frame()
        self._log_row_data("best",
                           self.time_best,
                           self.trace_best,
                           self.train_loss_best_str,
                           self.valid_loss_best_str,
                           None)
        self._print_low_frame()
        self._print_header()

    def _print_header(self):
        self._print_up_frame()
        self._print_row("       ",
                        "Train. time",
                        "Trace",
                        "Training loss  ",
                        "Valid.loss     ",
                        "TPS  ")
        self._print_mid_frame()

    def _log_row_data(self, name, secons, trace, train_loss_str, valid_loss_str, tps):
        seconds_str = format_seconds(secons)
        trace_str = format_trace_num(trace)
        tps_str = format_tps(tps) if tps is not None else ""

        if name == "current":
            line = '{0} │ {1} │ {2} │ {3} │ {4} │ {5}'.format(seconds_str, trace_str, train_loss_str,
                                                              self.train_loss_best_str, self.valid_loss_str, tps_str)
            self._logger.info(remove_non_ascii(line))
        name_str = name.capitalize()

        self._print_row(name_str, seconds_str, trace_str, train_loss_str, valid_loss_str, tps_str)

    def _print_row(self, name, time_row, trace_row, train_loss_row, valid_loss_row, tps_row):
        row = '{0:<7} │ {1:>{2}} │ {3:>{4}} │ {5:>15} │ {6:>15} │ {7:>5} '
        print(row.format(name,
                         time_row, self.time_row_width,
                         trace_row, self.trace_row_width,
                         train_loss_row,
                         valid_loss_row,
                         tps_row))

    def _print_up_frame(self):
        frame = '────────┬─{}─┬─{}─┬─────────────────┬─────────────────┬──────'
        print(frame.format('─'*self.time_row_width, '─'*self.trace_row_width))

    def _print_mid_frame(self):
        frame = '────────┼─{}─┼─{}─┼─────────────────┼─────────────────┼──────'
        print(frame.format('─'*self.time_row_width, '─'*self.trace_row_width))

    def _print_low_frame(self):
        frame = '────────┴─{}─┴─{}─┴─────────────────┴─────────────────┴──────'
        print(frame.format('─'*self.time_row_width, '─'*self.trace_row_width))

logger = Logger()
