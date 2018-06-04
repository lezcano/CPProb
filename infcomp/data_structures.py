from infcomp.logger import Logger


class SubBatch:
    def __init__(self, traces):
        self.traces = traces

    def sort_traces_observation(self):
        self.traces.sort(reverse=True, key=lambda t: len(t.observe))
        return self.traces

    def sort_traces_len(self):
        self.traces.sort(reverse=True, key=lambda trace: len(trace))
        return self.traces

    def example_trace(self):
        return self.traces[0]

    def get_indices(self, sub_batch):
        ret = []
        for trace in sub_batch.traces:
            for idx, trace_batch in enumerate(self.traces):
                if trace_batch.idx == trace.idx:
                    ret.append(idx)
                    continue
        return ret

    def __len__(self):
        return len(self.traces)


class Batch(SubBatch):
    def __init__(self, traces):
        super().__init__(traces)
        subbatches_dict = {}

        for trace in self.traces:
            if len(trace.samples) == 0:
                Logger.logger.log_error('Batch: Received a trace of length zero.')
            subbatches_dict.setdefault(trace.hash_address, []).append(trace)
        self.subbatches = [SubBatch(sub_batch) for sub_batch in subbatches_dict.values()]


class Trace:
    def __init__(self, samples, observe, idx):
        self.samples = samples
        self.observe = observe
        self.idx = idx
        self._hash_address = hash(tuple((sample.address for sample in self.samples)))

    @property
    def hash_address(self):
        return self._hash_address

    def __len__(self):
        return len(self.samples)


class Sample:
    def __init__(self, address, distr_fbb):
        self.address = address
        self.hash_address = hash(address)
        self.distr_fbb = distr_fbb


class Observe:
    def __init__(self, value):
        self.value = value

    def is_sequence(self):
        return self.value.dim() == 1

    def __len__(self):
        if self.is_sequence:
            return self.value.size()[0]
        raise RuntimeError("Tensor has no length.")
