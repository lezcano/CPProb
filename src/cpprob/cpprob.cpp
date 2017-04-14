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
Sample prev_sample;
Sample curr_sample;


void reset_trace(){
    t = Trace();
    prev_sample = Sample();
    curr_sample = Sample();
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
            int sample_instance = t.x_[id].size() + 1;

            t.samples_.emplace_back(Sample{addr, sample_instance, proposal<Distr>::type_enum,
                                           proposal<Distr>::request(Compilation::buff, distr),
                                           t.time_index_, x});
        }
    }
    else {
        static flatbuffers::FlatBufferBuilder foo;
        // Lua starts with 1
        int sample_instance = t.x_[id].size() + 1;
        curr_sample = Sample{addr, sample_instance, proposal<Distr>::type_enum,
                             proposal<Distr>::request(foo, distr)};

        auto proposal = Inference::get_proposal<Distr>(curr_sample, prev_sample);

        x = proposal(rng);
        prev_sample = curr_sample;
        prev_sample.set_value(std::exchange(x, distr(rng)));

        // Accumulate log(p/q) where q is the proposal distribution
        t.log_w_ += boost::math::pdf(math_distr(distr), x) - boost::math::pdf(math_distr(proposal), x);
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
