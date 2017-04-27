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
#include "cpprob/ndarray.hpp"

namespace cpprob {
template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = State::register_addr(addr);

    if (State::training) {
        x = distr(get_rng());

        if(from_observe){
            State::add_observe_to_batch(x);
        }
        else{
            State::add_sample_to_batch(Sample{addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum,
                                     default_type_param(distr), State::time_index(), x});
        }
    }
    else {
        State::curr_sample = Sample(addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum, default_type_param(distr));

        auto proposal = Inference::get_proposal<Distr, Params...>(State::curr_sample, State::prev_sample);

        x = proposal(get_rng());
        State::curr_sample.set_value(x);
        State::prev_sample = std::move(State::curr_sample);

        State::increment_cum_log_prob(pdf(distr, x) - pdf(proposal, x));
    }

    State::add_sample_to_trace(x, id);
    State::increment_time();

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr) {
    return sample_impl(distr, false);
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, typename Distr<Params ...>::result_type x) {
    if (State::training){
        sample_impl(distr, true);
    }
    else{
        using std::log;
        auto prob = cpprob::pdf(distr, x);
        State::add_observe_to_trace(prob);
        State::increment_cum_log_prob(prob);
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
    return std::vector<T>{static_cast<double>(args)...};
}

template<class Func, class... Args>
std::vector<std::vector<NDArray<double>>> inference(
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
        auto a = w*t;
        ret += a;
    }
    ret /= sum_w;
    return ret.x();
}
}       // namespace cpprob
#endif //INCLUDE_IMPL_CPPROB_HPP
