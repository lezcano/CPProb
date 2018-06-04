from glob import glob
import math
import numbers
from infcomp.logger import Logger
import sys
from infcomp.settings import settings

import torch
import torch.nn as nn
from torch.autograd import Variable


def file_starting_with(pattern, n):
    try:
        return sorted(glob(pattern + '*'))[n]
    except:
        Logger.logger.log_error('Cannot find file')
        sys.exit(1)


def load_nn(file_name):
    try:
        return torch.load(file_name).to(settings.device)
    except Exception:
        Logger.logger.log_error('Cannot load file {}'.format(file_name))
        raise


def save_if_its_time(nn, save_after_n_traces, n_processed_traces):
    # Note that if the size of the minibatch is very big and/or there are numbers in save_after_n_traces that are closer
    # this will be called at least two consecutive times. It is fine even if this happens
    if save_after_n_traces and n_processed_traces >= save_after_n_traces[-1]:
        nn.save_nn("{}_{}".format(nn.file_name, save_after_n_traces[-1]))
        save_after_n_traces.pop()
        if not save_after_n_traces:
            return True
    return False


def pad_sequence(sequences, batch_first=False, subbatches=False):
    # Modified from pytorch 0.4.0 to act on subbatches

    max_size = sequences[0].size()
    max_len, trailing_dims = max_size[0], max_size[1:]
    prev_l = max_len
    if subbatches:
        batch_size = sum(variable.size(1) for variable in sequences)
        trailing_dims = trailing_dims[1:]
    else:
        batch_size = len(sequences)
    if batch_first:
        out_dims = (batch_size, max_len) + trailing_dims
    else:
        out_dims = (max_len, batch_size) + trailing_dims

    out_variable = sequences[0].new_zeros(out_dims)
    n = 0
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        if subbatches:
            n_subbatch = variable.size(1)
        # temporary sort check, can be removed when we handle sorting internally
        if prev_l < length:
            raise ValueError("lengths array has to be sorted in decreasing order")
        prev_l = length
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            if subbatches:
                out_variable[n:n+n_subbatch, :length, ...] = variable
            else:
                out_variable[i, :length, ...] = variable
        else:
            if subbatches:
                out_variable[:length, n:n+n_subbatch, ...] = variable
            else:
                out_variable[:length, i, ...] = variable
        if subbatches:
            n += n_subbatch

    return out_variable


def logsumexp(value, dim=None, keepdim=False):
    # Taken from https://github.com/pytorch/pytorch/issues/2591#issuecomment-338980717
    """
    Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, numbers.Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    if isinstance(m, nn.Embedding):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
