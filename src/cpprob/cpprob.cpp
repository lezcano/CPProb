#include "cpprob/cpprob.hpp"

#include <string>
#include <unordered_map>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "cpprob/utils.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/trace.hpp"
#include "cpprob/socket.hpp"

namespace cpprob {

// Static variables
Trace t;
bool training;
boost::random::mt19937 rng{seeded_rng()};
std::unordered_map<std::string, int> ids;
PrevSampleInference prev_sample;
SampleInference curr_sample;


void reset_trace(){
    t = Trace();
    prev_sample = PrevSampleInference();
    curr_sample = SampleInference();
}

Trace get_trace(){ return t; }

void set_training(const bool t){
    training = t;
    reset_ids();
}

void reset_ids(){ ids.clear(); }

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = ids.emplace(addr, static_cast<int>(ids.size())).first->second;

    if (id >= static_cast<int>(t.x_.size()))
        t.x_.resize(id + 1);

    if (training) {
        x = distr(rng);

        if(from_observe){
            t.observes_.emplace_back(x);
        }
        else{
            // Lua starts with 1
            t.samples_.emplace_back(Sample{t.time_index_, static_cast<int>(t.x_[id].size()) + 1, x, distr_name<Distr>::value, addr});
        }
    }
    else {
        // Lua starts with 1
        int sample_instance = t.x_[id].size() + 1;
        curr_sample = SampleInference{addr, sample_instance, distr_name<Distr>::value};

        auto params = get_params(curr_sample, prev_sample);

        // TODO(Lezcano) Use last_x_ and id to compute x
        // x = sample_distr(posterior_distr(params));
        prev_sample = curr_sample;
        prev_sample.prev_sample_value = std::exchange(x, distr(rng));

        // TODO(Lezcano) Accumulate log(p/q) where q is the proposal distribution
    }

    t.x_[id].emplace_back(static_cast<double>(x));

    t.x_addr_.emplace_back(static_cast<double>(x), id);
    ++t.time_index_;

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr) {
    return sample_impl(distr, false);
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, double x) {
    if (training){
        sample_impl(distr, true);
    }
    else{
        using std::log;
        auto prob = pdf(math_distr(distr), x);
        t.y_.emplace_back(prob);
        t.log_w_ += log(prob);
    }
}

template double sample(boost::random::normal_distribution<double>&);

template void observe(boost::random::normal_distribution<double>&, double);
}  // namespace cpprob
