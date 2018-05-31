import logging
import cpuinfo
import time
import torch
import collections
import math
from termcolor import colored
from infcomp.settings import settings


def global_config():
    ret = ["PyTorch version : {}".format(torch.__version__)]
    cpu_info = cpuinfo.get_cpu_info()
    ret.append("CPU             : {}".format(cpu_info["brand"] if "brand" in cpu_info else "unknown"))
    ret.append("CPU count       : {}".format(cpu_info["count"] if "count" in cpu_info else "unknown"))
    ret.append("CUDA            : {}".format("available" if torch.cuda.is_available() else "not available"))
    if torch.cuda.is_available():
        ret.append("CUDA devices    : {0}".format(torch.cuda.device_count()))
        if settings.cuda_enabled:
            device_selected = settings.cuda_device != -1
            ret.append("CUDA selected   : {}".format(settings.cuda_device if device_selected else "all"))
    ret.append("Running on      : {}".format("CUDA" if settings.cuda_enabled else "CPU"))
    return "\n".join(ret)


def truncate_str(s, length=80):
    return (s[:length] + "...") if len(s) > length else s


def format_seconds(seconds):
    return time.strftime("%Hh %Mm %Ss", time.gmtime(seconds))


def format_trace_num(trace_num):
    # TODO(Lezcano) Simplify this format.format
    return "{:5}".format("{:,}".format(trace_num))


def format_tps(traces_per_sec):
    # TODO(Lezcano) Simplify this format.format
    return "{:3}".format("{:,}".format(int(traces_per_sec)))


def format_loss(loss, status):
    if status == "best":
        return colored("{:+.6e} ▼".format(loss), "green", attrs=["bold"])
    elif status == "better":
        return colored("{:+.6e}  ".format(loss), "green")
    elif status == "worse":
        return colored("{:+.6e}  ".format(loss), "red")
    else:
        raise NotImplementedError


class _Logger:
    def __init__(self, file_name="nn.log"):
        self._file_name = file_name

        self._logger = self.init_logger()

        # Fixme(Lezcano) Make dynamic
        self.time_row_width = 13
        self.trace_row_width = 7

        self.n_validation = 50
        self.validation_losses = collections.deque()
        self.sum_validation_losses = 0

        self.time_init = time.time()
        self.total_traces = 0

        self.time_last_batch = self.time_init

        self.train_loss_best = float("inf")
        self.validation_loss_best = float("inf")
        self.time_best_validation = self.time_init
        self.trace_best_validation = 0

    def init_logger(self):
        log = logging.getLogger()
        if self._file_name is not None:
            logger_file_handler = logging.FileHandler(self._file_name)
            logger_file_handler.setFormatter(logging.Formatter("%(asctime)s %(funcName)s: %(message)s"))
            log.addHandler(logger_file_handler)
        log.setLevel(logging.INFO)
        return log

    @property
    def total_time(self):
        return time.time() - self.time_init

    def log(self, line):
        print(line)
        self._logger.info(line)

    def log_error(self, line):
        trunc_line = truncate_str(line)
        formatted_line = colored("Error: {}".format(trunc_line), "red", attrs=["bold"])
        print(formatted_line)
        self._logger.error(line)

    def log_warning(self, line):
        trunc_line = truncate_str(line)
        formatted_line = colored("Warning: {}".format(trunc_line), "red", attrs=["bold"])
        print(formatted_line)
        self._logger.warning(line)

    def log_info(self, line):
        trunc_line = truncate_str(line)
        formatted_line = colored(trunc_line, "blue", attrs=["bold"])
        print(formatted_line)
        self._logger.info(line)

    def log_infer_begin(self, tcp_address):
        # No real need to log this
        print("")
        print(global_config())
        print("")
        print("Server connected to {}".format(tcp_address))

    def log_training_begin(self, param):
        self.log("")
        self.log(global_config())
        self.log("")
        self.log_info("Training from {}".format(param))
        self._print_header()

    def log_training(self, batch_length, train_loss, nn):
        now = time.time()
        time_spent_last_batch = now - self.time_last_batch
        traces_per_sec = batch_length / time_spent_last_batch
        self.time_last_batch = now

        self.total_traces += batch_length
        nn.total_traces = self.total_traces

        if len(self.validation_losses) > 0:
            previous_validation_loss = self.sum_validation_losses / len(self.validation_losses)
        else:
            previous_validation_loss = float("inf")

        # Update queue with last training losses (validation set)
        if not math.isnan(train_loss):
            if len(self.validation_losses) == self.n_validation:
                pop_loss = self.validation_losses.popleft()
                self.sum_validation_losses -= pop_loss
            self.validation_losses.append(train_loss)
            self.sum_validation_losses += train_loss
        validation_loss = self.sum_validation_losses / len(self.validation_losses)

        # Validation loss in green if it"s better than the last validation lost
        # If it"s the best validation loss, update the best time and best trace values
        if validation_loss < self.validation_loss_best:
            self.validation_loss_best = validation_loss
            nn.validation_loss_best = validation_loss
            self.time_best_validation = self.total_time
            self.trace_best_validation = self.total_traces
            status_val_loss = "best"
        elif validation_loss <= previous_validation_loss:
            status_val_loss = "better"
        else:
            status_val_loss = "worse"

        # Train loss in green if it"s better than the current validation loss
        if train_loss < self.train_loss_best:
            self.train_loss_best = train_loss
            nn.train_loss_best = train_loss
            status_train_loss = "best"
        elif train_loss <= validation_loss:
            status_train_loss = "better"
        else:
            status_train_loss = "worse"

        self._log_row_data("current",
                           self.total_time,
                           self.total_traces,
                           train_loss, status_train_loss,
                           validation_loss, status_val_loss,
                           traces_per_sec)

    def log_training_best(self):
        self._print_mid_frame()
        self._log_row_data("best",
                           self.time_best_validation,
                           self.trace_best_validation,
                           self.train_loss_best, "better",
                           self.validation_loss_best, "better")
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

    def _log_row_data(self, name, seconds, trace, train_loss, status_train_loss, valid_loss, status_val_loss, tps=None):
        seconds_str = format_seconds(seconds)
        trace_str = format_trace_num(trace)
        tps_str = format_tps(tps) if tps is not None else ""

        if name == "current":
            line = "{} │ {} │ {:+.6e} │ {:+.6e} │ {:+.6e}" \
                        .format(seconds_str, trace_str, train_loss, valid_loss, self.validation_loss_best)
            self._logger.info(line)
        name_str = name.capitalize()
        train_loss_str = format_loss(train_loss, status_train_loss)
        valid_loss_str = format_loss(valid_loss, status_val_loss)

        self._print_row(name_str, seconds_str, trace_str, train_loss_str, valid_loss_str, tps_str)

    def _print_row(self, name, time_row, trace_row, train_loss_row, valid_loss_row, tps_row):
        row = "{0:<7} │ {1:>{2}} │ {3:>{4}} │ {5:>15} │ {6:>15} │ {7:>5} "
        print(row.format(name,
                         time_row, self.time_row_width,
                         trace_row, self.trace_row_width,
                         train_loss_row,
                         valid_loss_row,
                         tps_row))

    def _print_up_frame(self):
        frame = "────────┬─{}─┬─{}─┬─────────────────┬─────────────────┬──────"
        print(frame.format("─"*self.time_row_width, "─"*self.trace_row_width))

    def _print_mid_frame(self):
        frame = "────────┼─{}─┼─{}─┼─────────────────┼─────────────────┼──────"
        print(frame.format("─"*self.time_row_width, "─"*self.trace_row_width))

    def _print_low_frame(self):
        frame = "────────┴─{}─┴─{}─┴─────────────────┴─────────────────┴──────"
        print(frame.format("─"*self.time_row_width, "─"*self.trace_row_width))


class Logger:
    logger = _Logger()

    @staticmethod
    def set(new_logger):
        Logger.logger = new_logger
        current_time = time.time()
        Logger.logger.time_init = current_time - (Logger.logger.time_last_batch - Logger.logger.time_init)
        Logger.logger.time_last_batch = current_time
        Logger.logger._logger = Logger.logger.init_logger()
