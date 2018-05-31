import zmq
import flatbuffers
import glob
import random
import time
import math

from infcomp.socket_wrapper import Socket
import infcomp.protocol.RequestTraces as RequestTraces
import infcomp.protocol.RequestFinishCompilation as RequestFinishCompilation
from infcomp.protocol.MessageBody import MessageBody
from infcomp.protocol.Message import Message
from infcomp.parse_flatbuffers import parse_message_body, parse_batch


class Requester:
    def minibatch(self, size):
        return self._get_traces(size)

    def _get_traces(self, n):
        raise NotImplementedError


class RequesterClient(Requester, Socket):
    def __init__(self, address):
        super().__init__(address, zmq.REQ)

    def _get_traces(self, n):
        builder = flatbuffers.Builder(64)
        RequestTraces.RequestTracesStart(builder)
        RequestTraces.RequestTracesAddNumTraces(builder, n)
        message_body = RequestTraces.RequestTracesEnd(builder)

        self._send_message(builder, message_body, MessageBody().RequestTraces)

        # Get response
        reply = self.receive()
        traces_fbb = parse_message_body(reply, MessageBody().ReplyTraces)
        return parse_batch([traces_fbb])

    def __exit__(self, exception_type, exception_value, traceback):
        self._finish_compilation()
        Socket.__exit__(self, exception_type, exception_value, traceback)

    def _finish_compilation(self):

        builder = flatbuffers.Builder(64)
        RequestFinishCompilation.RequestFinishCompilationStart(builder)
        message_body = RequestFinishCompilation.RequestFinishCompilationEnd(builder)

        self._send_message(builder, message_body, MessageBody().RequestFinishCompilation)

        # Discards Finish
        reply = self.receive()
        message = Message.GetRootAsMessage(reply, 0)
        if message.BodyType() != MessageBody().ReplyFinishCompilation:
            raise RuntimeError("Reply was not Finish Compilation.")


class RequesterFile(Requester):
    def __init__(self, traces_folder=None):
        self.traces_folder = traces_folder

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def _get_traces(self, n):
        """
        :param n: Number of elements in a batch.
        The current implementations takes the ceil(n/traces_per_file) as the actual batch size
        :return:
        """
        files = glob.glob("{}/*".format(self.traces_folder))
        traces_fbb_list = []
        # Take the first batch and check the size
        with open(random.choice(files), "rb") as f:
            reply = f.read()
            traces_fbb = parse_message_body(reply, MessageBody().ReplyTraces)
            traces_fbb_list.append(traces_fbb)
            n_files = int(math.ceil(n/traces_fbb.TracesLength()))
        file_names = [random.choice(files) for _ in range(n_files - 1)]

        for file_name in file_names:
            with open(file_name, "rb") as f:
                reply = f.read()
                traces_fbb = parse_message_body(reply, MessageBody().ReplyTraces)
                traces_fbb_list.append(traces_fbb)
        return parse_batch(traces_fbb_list)
