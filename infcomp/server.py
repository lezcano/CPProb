import zmq
import flatbuffers

from infcomp.socket_wrapper import Socket
from infcomp.parse_flatbuffers import parse_message_body
import infcomp.protocol.ReplyProposal as ReplyProposal
import infcomp.protocol.ReplyStartInference as ReplyStartInference
import infcomp.protocol.ReplyFinishInference as ReplyFinishInference
from infcomp.protocol.MessageBody import MessageBody
from infcomp.actions import parse_action
from infcomp.protocol.Distribution import Distribution


class Server(Socket):
    def __init__(self, address):
        super().__init__(address, zmq.REP)

    def get_action(self):
        message_body = parse_message_body(self.receive())
        return parse_action(message_body)

    def send_proposal(self, proposal, params):
        builder = flatbuffers.Builder(64)

        if proposal is None:
            distribution = None
            distribution_type = Distribution.NONE
        else:
            distribution, distribution_type = proposal.serialize(builder, params)

        ReplyProposal.ReplyProposalStart(builder)
        if distribution is not None:
            ReplyProposal.ReplyProposalAddDistribution(builder, distribution)
        ReplyProposal.ReplyProposalAddDistributionType(builder, distribution_type)
        message_body = ReplyProposal.ReplyProposalEnd(builder)

        self._send_message(builder, message_body, MessageBody().ReplyProposal)

    def reply_start_inference(self):
        builder = flatbuffers.Builder(64)
        ReplyStartInference.ReplyStartInferenceStart(builder)
        message_body = ReplyStartInference.ReplyStartInferenceEnd(builder)

        self._send_message(builder, message_body, MessageBody().ReplyStartInference)

    def reply_finish_inference(self):
        builder = flatbuffers.Builder(64)
        ReplyFinishInference.ReplyFinishInferenceStart(builder)
        message_body = ReplyFinishInference.ReplyFinishInferenceEnd(builder)

        self._send_message(builder, message_body, MessageBody().ReplyFinishInference)
