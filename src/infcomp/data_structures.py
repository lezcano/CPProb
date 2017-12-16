from infcomp.logger import logger

class Batch:
    def __init__(self, traces):
        self.batch = traces
        # Sub Batches are the batches with the same sequence of addresses
        # For now we will collapse them into a list, discarding the address
        self.sub_batches = {}
        self.len = len(traces)

        for trace in traces:
            if len(trace.samples) == 0:
                logger.log_error('Batch: Received a trace of length zero.')
            h = hash(trace.address())
            self.sub_batches.setdefault(h, []).append(trace)
        self.sub_batches = list(self.sub_batches.values())

    def __len__(self):
        return len(self.batch)

    @property
    def traces_lengths(self):
        return [len(t) for t in self.batch]

    def sort(self):
        # Sort the batch in decreasing trace length.
        self.batch = sorted(self.batch, reverse=True, key=lambda t: t.length)

    def cuda(self, device_id=None):
        for trace in self.batch:
            trace.cuda(device_id)

    def cpu(self):
        for trace in self.batch:
            trace.cpu()


class Trace:
    def __init__(self, samples, observe):
        self.samples = samples
        self.observe = observe

    def address(self):
        return '\n'.join(sample.address for sample in self.samples)

    def __len__(self):
        return len(self.samples)

    def cuda(self, device_id=None):
        self.observe.cuda(device_id)
        #TODO(Martínez) check I we have to transform anything more into tensors and send the to the gpu (addresses?)

    def cpu(self):
        self.observe.cpu()
        #TODO(Martínez) check I we have to transform anything more into tensors and send the to the cpu (addresses?)


class Sample:
    def __init__(self, address, distr_fbb):
        self.address = address
        self.distr_fbb = distr_fbb


class Observe:
    def __init__(self, ndarray):
        self.value = ndarray

    def cuda(self, device_id=None):
        self.value.cuda(device_id)

    def cpu(self):
        self.value.cpu()
