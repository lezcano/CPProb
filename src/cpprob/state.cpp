#include "cpprob/state.hpp"

#include <string>
#include <unordered_map>

#include "cpprob/trace.hpp"

namespace cpprob {

Trace State::t;
bool State::training;
std::unordered_map<std::string, int> State::ids_sample;
std::unordered_map<std::string, int> State::ids_predict;
Sample State::prev_sample;
Sample State::curr_sample;


void State::reset_trace()
{
    t = Trace();
    prev_sample = Sample();
    curr_sample = Sample();
}

Trace State::get_trace()
{
    return t;
}

void State::set_training(const bool t)
{
    training = t;
    reset_ids();
}

void State::reset_ids()
{
    ids_predict.clear();
    ids_sample.clear();
}

int State::sample_instance(int id)
{
    return State::t.sample_instance_[id];
}

int State::register_addr_sample(const std::string& addr)
{
    auto id = State::ids_sample.emplace(addr, static_cast<int>(State::ids_sample.size())).first->second;
    if (static_cast<int>(State::t.sample_instance_.size()) <= id){
        State::t.sample_instance_.resize(static_cast<size_t>(id) + 1);
    }
    // First id is 1
    // To change to zero change to post increment
    ++State::t.sample_instance_[id];
    return id;
}

void State::add_sample(const Sample& s)
{
    State::t.samples_.emplace_back(s);
}

void State::add_observe(const NDArray<double>& x)
{
    State::t.observes_.emplace_back(x);
}

int State::register_addr_predict(const std::string& addr)
{
    return State::ids_predict.emplace(addr, static_cast<int>(State::ids_predict.size())).first->second;
}

void State::add_predict(const std::string& addr, const NDArray<double>& x)
{
    auto id = register_addr_predict(addr);
    if (static_cast<int>(State::t.predict_.size()) >= id)
        State::t.predict_.resize(static_cast<size_t>(id) + 1);
    State::t.predict_[id].emplace_back(x);
    State::t.predict_addr_.emplace_back(id, x);
}

int State::time_index()
{
    return State::t.time_index_;
}

void State::increment_time()
{
    ++State::t.time_index_;
}
void State::increment_cum_log_prob(double log_p){
    State::t.log_w_ += log_p;
}

}  // namespace cpprob
