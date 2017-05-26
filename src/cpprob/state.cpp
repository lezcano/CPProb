#include "cpprob/state.hpp"

#include <string>
#include <unordered_map>


#include "cpprob/any.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/trace.hpp"

namespace cpprob {

TraceCompile State::t_comp;
TracePredicts State::t_pred;
StateType State::state;
std::unordered_map<std::string, int> State::ids_sample;
std::unordered_map<std::string, int> State::ids_predict;
Sample State::prev_sample;
Sample State::curr_sample;


void State::reset_trace()
{
    t_comp = TraceCompile();
    t_pred = TracePredicts();
    prev_sample = Sample();
    curr_sample = Sample();
}

TraceCompile State::get_trace_comp()
{
    return t_comp;
}

TracePredicts State::get_trace_pred()
{
    return t_pred;
}

void State::set(StateType s)
{
    state = s;
    reset_ids();
}

StateType State::current_state()
{
    return state;
}

void State::reset_ids()
{
    ids_predict.clear();
    ids_sample.clear();
}

int State::sample_instance(int id)
{
    return State::t_comp.sample_instance_[id];
}

int State::register_addr_sample(const std::string& addr)
{
    auto id = State::ids_sample.emplace(addr, static_cast<int>(State::ids_sample.size())).first->second;
    if (static_cast<int>(State::t_comp.sample_instance_.size()) <= id){
        State::t_comp.sample_instance_.resize(static_cast<size_t>(id) + 1);
    }
    // First id is 1
    // To change to zero change to post increment
    ++State::t_comp.sample_instance_[id];
    return id;
}

void State::add_sample(const Sample& s)
{
    State::t_comp.samples_.emplace_back(s);
}

void State::add_observe(const NDArray<double>& x)
{
    State::t_comp.observes_.emplace_back(x);
}

int State::time_index()
{
    return State::t_comp.time_index_;
}

void State::increment_time()
{
    ++State::t_comp.time_index_;
}

int State::register_addr_predict(const std::string& addr)
{
    return State::ids_predict.emplace(addr, static_cast<int>(State::ids_predict.size())).first->second;
}

void State::add_predict(const std::string& addr, const cpprob::any &x)
{
    auto id = register_addr_predict(addr);
    State::t_pred.add_predict(id, x);
}

void State::increment_cum_log_prob(double log_p)
{
    State::t_pred.increment_cum_log_prob(log_p);
}

void State::serialize_ids_pred(std::ofstream & out_file)
{
    using namespace detail;
    std::vector<std::string> addresses(State::ids_predict.size());
    for(const auto& kv : State::ids_predict)
        addresses[kv.second] = kv.first;
    out_file << addresses;
}

}  // namespace cpprob
