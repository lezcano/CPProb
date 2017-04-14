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


void State::reset_trace(){
    t = Trace();
    prev_sample = Sample();
    curr_sample = Sample();
}

Trace State::get_trace(){ return t; }

void State::set_training(const bool t){
    training = t;
    reset_ids();
}

void State::reset_ids(){ ids.clear(); }

}  // namespace cpprob
