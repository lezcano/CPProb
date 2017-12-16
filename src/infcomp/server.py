import zmq
import flatbuffers

from infcomp.socket_wrapper import Socket
from infcomp.parse_flatbuffers import parse_message_body
import infcomp.protocol.ReplyProposal as ReplyProposal
import infcomp.protocol.Message as Message
import infcomp.protocol.ReplyStartInference as ReplyStartInference
from infcomp.protocol.MessageBody import MessageBody
from infcomp.actions import parse_action


class Server(Socket):
    def __init__(self, address):
        super().__init__(address, zmq.REP)

    def get_action(self):
        message_body = parse_message_body(self.receive())
        return parse_action(message_body)

    def send_proposal(self, proposal, params):
        builder = flatbuffers.Builder(64)
        distribution, distribution_type = proposal.serialize(builder, params)

        ReplyProposal.ReplyProposalStart(builder)
        ReplyProposal.ReplyProposalAddDistribution(builder, distribution)
        ReplyProposal.ReplyProposalAddDistributionType(builder, distribution_type)
        message_body = ReplyProposal.ReplyProposalEnd(builder)

        self._send_message(builder, message_body, MessageBody().ReplyProposal)

    def reply_start_inference(self):
        builder = flatbuffers.Builder(64)
        ReplyStartInference.ReplyStartInferenceStart(builder)
        message_body = ReplyStartInference.ReplyStartInferenceEnd(builder)

        self._send_message(builder, message_body, MessageBody().ReplyStartInference)

    def _send_message(self, builder, message_body,  message_type):
        Message.MessageStart(builder)
        Message.MessageAddBodyType(builder, message_type)
        Message.MessageAddBody(builder, message_body)
        message = Message.MessageEnd(builder)
        builder.Finish(message)
        message = builder.Output()

        self.send(message)
