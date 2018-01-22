#include "cpprob/state.hpp"

#include <cstdio>             // for remove
#include <iterator>           // for make_move_iterator, move_iterator, oper...
#include <numeric>            // for accumulate
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
        }
        else if (state_ == StateType::csis) {
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

bool State::csis ()
{
    return state_ == StateType::csis;
}

bool State::sis ()
{
    return state_ == StateType::sis;
}

bool State::dryrun ()
{
    return state_ == StateType::dryrun;
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

void StateCompile::add_observe(NDArray<double>&& x)
{
    StateCompile::trace_.observes_.emplace_back(std::move(x));
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
std::map<std::string, double> StateInfer::log_prob_rej_samp_;
bool StateInfer::all_int_empty = true;
bool StateInfer::all_real_empty = true;
bool StateInfer::all_any_empty = true;
boost::filesystem::path StateInfer::dump_file_;
std::vector<void (*)()> StateInfer::clear_functions_;

void StateInfer::start_infer()
{
    TraceInfer::ids_predict_.clear();
    clear_empty_flags();
}


void StateInfer::finish_infer()
{
    dump_ids(TraceInfer::ids_predict_, get_file_name("ids"));
    if (all_int_empty) {
        std::remove(get_file_name("int").c_str());
    }
    if (all_real_empty) {
        std::remove(get_file_name("real").c_str());
    }
    if (all_any_empty) {
        std::remove(get_file_name("any").c_str());
    }
    clear_empty_flags();
    if (State::csis()) {
        SocketInfer::send_finish_inference();
    }
}


void StateInfer::add_value_to_sample(const boost::any & x)
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
    StateInfer::dump_predicts(trace_.predict_int_, trace_.log_w_, get_file_name("int"));
    StateInfer::dump_predicts(trace_.predict_real_, trace_.log_w_, get_file_name("real"));
    StateInfer::dump_predicts(trace_.predict_any_, trace_.log_w_, get_file_name("any"));
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


void StateInfer::increment_log_prob(const double log_p, const std::string & addr)
{
    // If the address is empty it's because it's an observe. Probs a variant here would be better...

    if (State::csis() &&
        State::rejection_sampling() &&
        !addr.empty()) {
        log_prob_rej_samp_[addr] = log_p;
    }
    else {
        trace_.log_w_ += log_p;
    }
}

void StateInfer::finish_rejection_sampling()
{
    const auto log_p = std::accumulate(std::begin(log_prob_rej_samp_),
                                       std::end(log_prob_rej_samp_),
                                       0.0,
                                       [](const double previous,
                                          const auto & p) { return previous + p.second; });
    trace_.log_w_ += log_p;
    for(const auto & f : clear_functions_) {
        f();
    }
    clear_functions_.clear();
    log_prob_rej_samp_.clear();
}

void StateInfer::config_file(const boost::filesystem::path & dump_file)
{
    dump_file_ = dump_file;
}

boost::filesystem::path StateInfer::get_file_name(const std::string & value)
{
    std::string mode = []{
        if      (State::csis()) { return "_csis"; }
        else if (State::sis())  { return "_sis"; }
        else { throw std::runtime_error("Unsupported mode."); }
    }();
    return dump_file_.string() + mode + '.' + value;
}

void StateInfer::dump_ids(const std::unordered_map<std::string, int> & ids_predict, const boost::filesystem::path & path)
{
    std::ofstream f{path.c_str()};
    std::vector<std::string> addresses(ids_predict.size());
    for(const auto & kv : ids_predict) {
        addresses[kv.second] = kv.first;
    }
    for(const auto & address : addresses) {
        f << address << std::endl;
    }
}

void StateInfer::dump_predicts(const std::vector<std::pair<int, cpprob::any>> & predicts, const double log_w, const boost::filesystem::path & path)
{
    std::ofstream f{path.c_str(), std::ios::app};
    f.precision(std::numeric_limits<double>::digits10);
    f << std::scientific << std::make_pair(predicts, log_w) << std::endl;
}

}  // namespace cpprob
