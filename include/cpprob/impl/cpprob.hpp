#ifndef INCLUDE_IMPL_CPPROB_HPP
#define INCLUDE_IMPL_CPPROB_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <string>

#include "cpprob/utils.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/state.hpp"
#include "cpprob/socket.hpp"

namespace cpprob {
template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = State::ids.emplace(addr, static_cast<int>(State::ids.size())).first->second;

    if (id >= static_cast<int>(State::t.x_.size()))
        State::t.x_.resize(id + 1);

    if (State::training) {
        x = distr(get_rng());

        if(from_observe){
            State::t.observes_.emplace_back(x);
        }
        else{
            int sample_instance = State::t.x_[id].size() + 1;

            State::t.samples_.emplace_back(Sample{addr, sample_instance, proposal<Distr>::type_enum,
                                           proposal<Distr>::request(Compilation::buff, distr),
                                           State::t.time_index_, x});
        }
    }
    else {
        static flatbuffers::FlatBufferBuilder foo;
        // Lua starts with 1
        int sample_instance = State::t.x_[id].size() + 1;
        State::curr_sample = Sample{addr, sample_instance, proposal<Distr>::type_enum,
                             proposal<Distr>::request(foo, distr)};

        auto proposal = Inference::get_proposal<Distr>(State::curr_sample, State::prev_sample);

        x = proposal(get_rng());
        State::prev_sample = State::curr_sample;
        State::prev_sample.set_value(std::exchange(x, distr(get_rng())));

        // Accumulate log(p/q) where q is the proposal distribution
        State::t.log_w_ += boost::math::pdf(math_distr(distr), x) - boost::math::pdf(math_distr(proposal), x);
    }

    State::t.x_[id].emplace_back(static_cast<double>(x));

    State::t.x_addr_.emplace_back(static_cast<double>(x), id);
    ++State::t.time_index_;

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr) {
    return sample_impl(distr, false);
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, double x) {
    if (State::training){
        sample_impl(distr, true);
    }
    else{
        using std::log;
        auto prob = pdf(math_distr(distr), x);
        State::t.y_.emplace_back(prob);
        State::t.log_w_ += log(prob);
    }
}

template<class Func>
void compile(const Func& f, const std::string& tcp_addr) {
    Compilation::connect_server(tcp_addr);
    State::set_training(true);

    while(true){
        auto batch_size = Compilation::get_batch_size();

        for (int i = 0; i < batch_size; ++i){
            State::reset_trace();
            f();
            Compilation::add_trace(State::get_trace());
        }
        Compilation::send_batch();
    }
}


template<class T, class... Args>
std::vector<T> embed_observe(Args&&... args){
    return std::vector<T>{args...};
}

template<class Func, class... Args>
std::vector<std::vector<double>> inference(
        const Func& f,
        const std::string& tcp_addr,
        size_t n,
        Args&&... args){

    Inference::connect_client(tcp_addr);
    State::set_training(false);

    Inference::send_observe_init(embed_observe<double>(args...));

    double sum_w = 0;
    Trace ret;
    for (size_t i = 0; i < n; ++i) {
        State::reset_trace();
        f(args...);
        auto t = State::get_trace();
        auto w = std::exp(t.log_w());
        sum_w += w;
        ret += w*t;
    }
    ret /= sum_w;
    return ret.x();
}
}       // namespace cpprob
#endif //INCLUDE_IMPL_CPPROB_HPP
