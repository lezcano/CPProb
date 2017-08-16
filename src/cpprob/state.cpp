#include "cpprob/state.hpp"

#include <iomanip>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>


#include "cpprob/any.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/socket.hpp"
#include "cpprob/trace.hpp"

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

void StateCompile::start_batch()
{
    batch_.clear();
}

void StateCompile::finish_batch()
{
    SocketCompile::send_batch(batch_);
}

void StateCompile::start_trace()
{
    trace_ = TraceCompile();
}

void StateCompile::finish_trace()
{
    batch_.emplace_back(std::move(trace_));
}

void StateCompile::add_sample(const Sample& s)
{
    if (State::rejection_sampling()) {
        StateCompile::trace_.samples_rejection_.emplace_back(s);
    }
    else {
        StateCompile::trace_.samples_.emplace_back(s);
    }
}

int StateCompile::sample_instance(const std::string & addr)
{
    // ids start in 1
    return ++StateCompile::trace_.sample_instance_[addr];
}

int StateCompile::time_index()
{
    return StateCompile::trace_.time_index_;
}

void StateCompile::increment_time()
{
    ++StateCompile::trace_.time_index_;
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
        auto emplaced = samples_processed.emplace(it->sample_address());
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
Sample StateInfer::prev_sample_;
Sample StateInfer::curr_sample_;
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


void StateInfer::start_trace()
{
    prev_sample_ = Sample();
    curr_sample_ = Sample();
    trace_ = TraceInfer();
}

void StateInfer::finish_trace()
{
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
