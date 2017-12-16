import zmq
import flatbuffers

from infcomp.socket_wrapper import Socket
import infcomp.protocol.RequestTraces as RequestTraces
import infcomp.protocol.Message as Message
import infcomp.protocol.MessageBody as MessageBody
from infcomp.parse_flatbuffers import parse_message_body, parse_batch


class Client(Socket):
    def __init__(self, address):
        super().__init__(address, zmq.REQ)

    def validation_set(self, size):
        return self._get_traces(size)

    def minibatch(self, size):
        return self._get_traces(size)

    def _get_traces(self, n):
        # allocate buffer
        builder = flatbuffers.Builder(64)

        # construct message body
        RequestTraces.RequestTracesStart(builder)
        RequestTraces.RequestTracesAddNumTraces(builder, n)
        message_body = RequestTraces.RequestTracesEnd(builder)

        # construct message
        Message.MessageStart(builder)
        Message.MessageAddBodyType(builder, MessageBody.MessageBody().RequestTraces)
        Message.MessageAddBody(builder, message_body)
        message = Message.MessageEnd(builder)
        builder.Finish(message)
        message = builder.Output()

        # Send message
        self.send(message)

        # Get response
        reply = self.receive()
        traces_fbb = parse_message_body(reply, MessageBody.MessageBody().ReplyTraces)
        return parse_batch(traces_fbb)
