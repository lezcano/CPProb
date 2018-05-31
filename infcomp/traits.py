from infcomp.protocol.MessageBody import MessageBody
from infcomp.protocol.Distribution import Distribution

from infcomp.protocol.ReplyTraces import ReplyTraces
from infcomp.protocol.ReplyFinishCompilation import ReplyFinishCompilation
from infcomp.protocol.RequestStartInference import RequestStartInference
from infcomp.protocol.RequestFinishInference import RequestFinishInference
from infcomp.protocol.RequestProposal import RequestProposal

from infcomp.protocol.Normal import Normal
from infcomp.protocol.Discrete import Discrete
from infcomp.protocol.UniformDiscrete import UniformDiscrete
from infcomp.protocol.UniformContinuous import UniformContinuous

# Dynamic type-traits: Enum to class methods


def message_body_class(msg):
    message_body_class.dict = {MessageBody.ReplyTraces: ReplyTraces,
                               MessageBody.ReplyFinishCompilation: ReplyFinishCompilation,
                               MessageBody.RequestStartInference: RequestStartInference,
                               MessageBody.RequestFinishInference: RequestFinishInference,
                               MessageBody.RequestProposal: RequestProposal,
                               MessageBody.NONE: None}
    return message_body_class.dict[msg]


def distribution_class(msg):
    distribution_class.dict = {Distribution.Normal: Normal,
                               Distribution.Discrete: Discrete,
                               Distribution.UniformDiscrete: UniformDiscrete,
                               Distribution.UniformContinuous: UniformContinuous,
                               Distribution.NONE: None}
    return distribution_class.dict[msg]
