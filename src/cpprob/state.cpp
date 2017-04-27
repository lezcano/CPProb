#include "cpprob/state.hpp"

#include <string>
#include <unordered_map>

#include "cpprob/trace.hpp"

namespace cpprob {

Trace State::t;
bool State::training;
std::unordered_map<std::string, int> State::ids;
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
    ids.clear();
}

int State::sample_instance(int id)
{
    // Lua starts with 1
    return State::t.x_[id].size() + 1;
}

int State::register_addr(const std::string& addr)
{
    auto id = State::ids.emplace(addr, static_cast<int>(State::ids.size())).first->second;
    // Might not be the first execution and maybe ids is already populated and we have to increase
    // the size of x_ by more than one
    if (id >= static_cast<int>(State::t.x_.size()))
        State::t.x_.resize(id + 1);
    return id;
}

void State::add_sample_to_batch(const Sample& s)
{
    State::t.samples_.emplace_back(s);
}

void State::add_observe_to_batch(const NDArray<double>& n)
{
    State::t.observes_.emplace_back(n);
}

void State::add_sample_to_trace(const NDArray<double>& x, int id)
{
    State::t.x_[id].emplace_back(x);
    State::t.x_addr_.emplace_back(x, id);
}

void State::add_observe_to_trace(double prob)
{
    State::t.y_.emplace_back(prob);
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
