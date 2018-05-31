import flatbuffers
import numpy as np

from infcomp.settings import settings

from infcomp.protocol.Message import Message

import infcomp.protocol.NDArray

import infcomp.protocol.Normal
import infcomp.protocol.Discrete
import infcomp.protocol.UniformDiscrete
import infcomp.protocol.UniformContinuous
from infcomp.distributions.prior_distributions import PriorNormal
from infcomp.distributions.prior_distributions import PriorUniformDiscrete
from infcomp.distributions.prior_distributions import PriorDiscrete
from infcomp.distributions.prior_distributions import PriorUniformContinuous

from infcomp.data_structures import Sample, Observe, Trace, Batch
from infcomp.traits import distribution_class, message_body_class
from infcomp.logger import Logger


def distributionfbb_prior(distribution_fbb):
    # TODO(Lezcano) Generalise with getattr
    distributionfbb_prior.dict = {infcomp.protocol.Normal.Normal: PriorNormal,
                                  infcomp.protocol.UniformDiscrete.UniformDiscrete: PriorUniformDiscrete,
                                  infcomp.protocol.Discrete.Discrete: PriorDiscrete,
                                  infcomp.protocol.UniformContinuous.UniformContinuous: PriorUniformContinuous,
                                  None: None}
    try:
        return distributionfbb_prior.dict[distribution_fbb]
    except KeyError:
        Logger.logger.log_error('Unsupported distribution: ' + distribution_fbb)
        raise


def parse_observation(observation_fbb):
    if observation_fbb is None:
        return Observe(settings.Tensor())
    # TODO(Lezcano) This method will be deprecated in the next flatbuffers release (1.8.0)
    # data = observation_fbb.dataAsNumpy()
    # shape = observation_fbb.shapeAsNumpy()
    b = observation_fbb._tab.Bytes

    o = flatbuffers.number_types.UOffsetTFlags.py_type(observation_fbb._tab.Offset(4))
    offset = observation_fbb._tab.Vector(o) if o != 0 else 0
    length = observation_fbb.DataLength()
    data_np = np.frombuffer(b, offset=offset, dtype=np.dtype('float64'), count=length)

    o = flatbuffers.number_types.UOffsetTFlags.py_type(observation_fbb._tab.Offset(6))
    offset = observation_fbb._tab.Vector(o) if o != 0 else 0
    length = observation_fbb.ShapeLength()
    shape_np = np.frombuffer(b, offset=offset, dtype=np.dtype('int32'), count=length)

    data = data_np.reshape(shape_np)
    return Observe(settings.Tensor(data))


def parse_sample(sample_fbb):
    # Address
    address = sample_fbb.Address().decode("utf-8")

    # Prior
    distr_fbb_class = distribution_class(sample_fbb.DistributionType())
    if distr_fbb_class is not None:
        distr_fbb = distr_fbb_class()
        distr_fbb.Init(sample_fbb.Distribution().Bytes, sample_fbb.Distribution().Pos)
    else:
        distr_fbb = None

    return Sample(address, distr_fbb)


def parse_trace(trace_fbb, idx):
    obs = parse_observation(trace_fbb.Observation())

    samples = []
    for j in range(trace_fbb.SamplesLength()):
        sample_fbb = trace_fbb.Samples(j)
        sample = parse_sample(sample_fbb)
        samples.append(sample)

    return Trace(samples, obs, idx)


def parse_batch(traces_fbb):
    """
    :param traces_fbb: list of traces objects from flatbuffers
    """
    i = 0
    traces_list = []
    for traces_fbb_one in traces_fbb:
        for j in range(traces_fbb_one.TracesLength()):
            traces_list.append(parse_trace(traces_fbb_one.Traces(j), i))
            i += 1
    return Batch(traces_list)


def parse_message_body(message, type_message=None):
    message = Message.GetRootAsMessage(message, 0)
    if type_message is not None and message.BodyType() != type_message:
        raise RuntimeError(
            "parse_message_body: Unexpected body: MessageBody id: {}. Requested {}".format(message.BodyType(), type_message))
    message_body = message_body_class(message.BodyType())()
    # Handles messages with empty body
    if hasattr(message.Body(), "Bytes"):
        message_body.Init(message.Body().Bytes, message.Body().Pos)
    return message_body
