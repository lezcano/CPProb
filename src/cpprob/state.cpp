#include "cpprob/state.hpp"

#include <iterator>           // for make_move_iterator, move_iterator, oper...
#include <set>                // for set
#include <string>             // for string
#include <unordered_map>      // for unordered_map

#include "cpprob/socket.hpp"  // for SocketInfer, SocketCompile
#include "cpprob/trace.hpp"   // for TraceCompile, TraceInfer, TraceInfer::i...

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
//////////////////////////          State             //////////////////////////
////////////////////////////////////////////////////////////////////////////////

StateType State::state_;
bool State::rejection_sampling_ = false;

void State::set(StateType s)
{
    state_ = s;
}

void State::start_rejection_sampling()
{
    if(state_ != StateType::dryrun) {
        rejection_sampling_ = true;
    }
}

void State::finish_rejection_sampling()
{
    if(state_ != StateType::dryrun) {
        rejection_sampling_ = false;
        if (state_ == StateType::compile) {
            StateCompile::finish_rejection_sampling();
        } else if (state_ == StateType::inference) {
            StateInfer::finish_rejection_sampling();
        }
    }
}

bool State::rejection_sampling()
{
    return rejection_sampling_;
}

bool State::compile ()
{
    return state_ == StateType::compile;
}

bool State::inference ()
{
    return state_ == StateType::inference ||
           state_ == StateType::importance_sampling;

}

StateType State::state()
{
    return state_;
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

TraceCompile StateCompile::trace_;
std::vector<TraceCompile> StateCompile::batch_;
flatbuffers::FlatBufferBuilder StateCompile::buff_;

void StateCompile::start_batch()
{
    batch_.clear();
}

void StateCompile::finish_batch()
{
    std::vector<flatbuffers::Offset<protocol::Trace>> fbb_traces;
    std::transform(batch_.begin(), batch_.end(), std::back_inserter(fbb_traces),
                   [](const TraceCompile & trace) { return trace.pack(buff_); });

    auto traces = protocol::CreateReplyTracesDirect(buff_, &fbb_traces);
    auto msg = protocol::CreateMessage(
            buff_,
            protocol::MessageBody::ReplyTraces,
            traces.Union());
    buff_.Finish(msg);
    SocketCompile::send_batch(buff_);
    buff_.Clear();
}

void StateCompile::start_trace()
{
    trace_ = TraceCompile();
}

void StateCompile::finish_trace()
{
    batch_.emplace_back(std::move(trace_));
}

void StateCompile::add_observe(const NDArray<double>& x)
{
    StateCompile::trace_.observes_.emplace_back(x);
}

// Accept / Reject sampling
void StateCompile::finish_rejection_sampling()
{
    std::set<std::string> samples_processed;
    std::vector<Sample> last_samples;
    for (auto it = trace_.samples_rejection_.rbegin(); it != trace_.samples_rejection_.rend(); ++it) {
        auto emplaced = samples_processed.emplace(it->address());
        // If it is the first time that we find that sample
        if (emplaced.second) {
            last_samples.emplace_back(std::move(*it));
        }
    }
    trace_.samples_rejection_.clear();
    // TODO(Lezcano) Replace with resize + several moves
    for (auto it = std::make_move_iterator(last_samples.rbegin());
         it != std::make_move_iterator(last_samples.rend());
         ++it) {
        trace_.samples_.emplace_back(*it);
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////         Inference             ////////////////////////
////////////////////////////////////////////////////////////////////////////////

TraceInfer StateInfer::trace_;
flatbuffers::FlatBufferBuilder StateInfer::buff_;
bool StateInfer::all_int_empty = true;
bool StateInfer::all_real_empty = true;
bool StateInfer::all_any_empty = true;
std::vector<void (*)()> StateInfer::clear_functions_;

void StateInfer::start_infer()
{
    TraceInfer::ids_predict_.clear();
    clear_empty_flags();
}


void StateInfer::finish_infer()
{
    SocketInfer::dump_ids(TraceInfer::ids_predict_);
    if (all_int_empty) {
        SocketInfer::delete_file("_int");
    }
    if (all_real_empty) {
        SocketInfer::delete_file("_real");
    }
    if (all_any_empty) {
        SocketInfer::delete_file("_any");
    }
    clear_empty_flags();
}


void StateInfer::add_value_to_sample(const NDArray<double> & x)
{
    trace_.curr_sample_.set_value(x);
}

void StateInfer::start_trace()
{
    trace_ = TraceInfer();
}

void StateInfer::finish_trace()
{
    buff_.Clear();
    SocketInfer::dump_predicts(trace_.predict_int_, trace_.log_w_, "_int");
    SocketInfer::dump_predicts(trace_.predict_real_, trace_.log_w_, "_real");
    SocketInfer::dump_predicts(trace_.predict_any_, trace_.log_w_, "_any");
    all_int_empty &= trace_.predict_int_.size() == 0;
    all_real_empty &= trace_.predict_real_.size() == 0;
    all_any_empty &= trace_.predict_any_.size() == 0;
}

void StateInfer::clear_empty_flags()
{
    all_int_empty = true;
    all_real_empty = true;
    all_any_empty = true;
}


void StateInfer::increment_log_prob(const double log_p)
{
    trace_.log_w_ += log_p;
}

void StateInfer::finish_rejection_sampling()
{
    for(const auto & f : clear_functions_) {
        f();
    }
    clear_functions_.clear();
}

}  // namespace cpprob
