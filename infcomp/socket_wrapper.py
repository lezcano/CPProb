import zmq
from infcomp.logger import Logger
import infcomp.protocol.Message as Message


# TODO Improve error handling, if any.
# This class should be used either in a `with Socket(addr)...` clause or
#  the user should remember to connect and close the socket herself
class Socket:
    def __init__(self, addr, mode):
        self._context = zmq.Context()
        self._socket = self._context.socket(mode)
        self._mode = mode
        self._addr = addr

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def connect(self):
        # Server
        if self._mode == zmq.REQ:
            self._socket.connect(self._addr)
        # Client
        elif self._mode == zmq.REP:
            self._socket.bind(self._addr)
        else:
            raise RuntimeError("Wrong socket mode {}. ".format(self._mode))

    def close(self):
        self._socket.close()
        self._context.term()

    def send(self, request):
        try:
            self._socket.send(request)
        except zmq.error.ZMQError:
            print("Could not send the Message. Is CPProb listening?")

    def receive(self):
        return self._socket.recv()

    def _send_message(self, builder, message_body,  message_type):
        Message.MessageStart(builder)
        Message.MessageAddBodyType(builder, message_type)
        Message.MessageAddBody(builder, message_body)
        message = Message.MessageEnd(builder)
        builder.Finish(message)
        message = builder.Output()

        self.send(message)
