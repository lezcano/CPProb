from infcomp.parse_flatbuffers import parse_sample, parse_observation

import infcomp.protocol.RequestStartInference
import infcomp.protocol.RequestFinishInference
import infcomp.protocol.RequestProposal


def parse_action(action_fbb):
    # TODO(Lezcano) Generalise with getattr
    parse_action.dict = {infcomp.protocol.RequestStartInference.RequestStartInference: StartInference,
                         infcomp.protocol.RequestProposal.RequestProposal: GetProposal,
                         infcomp.protocol.RequestFinishInference.RequestFinishInference: FinishInference}
    return parse_action.dict[type(action_fbb)](action_fbb)


class Action:
    def run(self, nn, server, file):
        raise NotImplementedError


class StartInference(Action):
    def __init__(self, start_inference_fbb):
        self._observation = parse_observation(start_inference_fbb.Observation())

    def run(self, nn, server, file=None):
        # Notify that the inference is ready to start
        server.reply_start_inference()
        nn.set_observation(self._observation)


class FinishInference(Action):
    def __init__(self, finish_inference_fbb):
        pass

    def run(self, nn, server, file=None):
        server.reply_finish_inference()


class GetProposal(Action):
    def __init__(self, get_proposal_fbb):
        self._current_sample = parse_sample(get_proposal_fbb.CurrentSample())
        self._previous_sample = parse_sample(get_proposal_fbb.PreviousSample())

    def run(self, nn, server, file):
        proposal, params = nn.get_proposal(self._current_sample, self._previous_sample)
        if file is not None:
            # TODO(Martinez) should we format this better?
            file.write(str(params))
        server.send_proposal(proposal, params)
