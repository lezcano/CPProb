#ifndef INCLUDE_CPPROB_HPP
#define INCLUDE_CPPROB_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <string>

#include "cpprob/utils.hpp"
#include "cpprob/trace.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/state.hpp"
#include "cpprob/socket.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {
namespace detail {

template<class T, class U, class = std::enable_if_t<std::is_arithmetic<U>::value>>
std::vector<T> to_vec(U value)
{
    return std::vector<T>({static_cast<T>(value)});
}

template<class T, class U, class V>
std::vector<T> to_vec(const std::pair<U, V>& pair);

template<class T, class U>
std::vector<T> to_vec(const std::vector<U>& v)
{
    std::vector<T> ret;
    for(const auto& elem : v) {
        auto aux = to_vec<T>(elem);
        ret.insert(ret.end(), aux.begin(), aux.end());

    }
    return ret;
}

template<class T, class U, class V>
std::vector<T> to_vec(const std::pair<U, V>& pair)
{
    auto ret = to_vec<T>(pair.first);
    auto aux = to_vec<T>(pair.second);

    ret.insert(ret.end(), aux.begin(), aux.end());
    return ret;
}

template<class T, class... Args, size_t... Indices>
std::vector<T> to_vec_tuple(const std::tuple<Args...>& tup, std::index_sequence<Indices...>)
{
    std::vector<T> ret;
    auto append_ret = [&](std::vector<T>&& other) { ret.insert(ret.end(), std::make_move_iterator(other.begin()),
                                                                      std::make_move_iterator(other.end())); };
    (void)std::initializer_list<int>{ (append_ret(to_vec<T>(std::get<Indices>(tup))), 0)... };
    return ret;
}

template<class T, class... Args>
std::vector<T> to_vec(const std::tuple<Args...>& args){
    return detail::to_vec_tuple<T, Args...>(args, std::make_index_sequence<sizeof...(Args)>());
}

}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = State::register_addr_sample(addr);

    if (State::training) {
        x = distr(get_rng());

        if(from_observe){
            State::add_observe(x);
        }
        else{
            State::add_sample(Sample{addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum,
                                     default_distr(distr), State::time_index(), x});
        }
    }
    else {
        State::curr_sample = Sample(addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum, default_distr(distr));

        auto proposal = Inference::get_proposal<Distr, Params...>(State::curr_sample, State::prev_sample);

        x = proposal(get_rng());
        State::curr_sample.set_value(x);
        State::prev_sample = std::move(State::curr_sample);

        State::increment_cum_log_prob(pdf(distr, x) - pdf(proposal, x));
    }

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
        State::increment_cum_log_prob(prob);
    }
}

void predict(const NDArray<double> &x);

template<class Func>
void compile(const Func& f, const std::string& tcp_addr) {
    Compilation::connect_server(tcp_addr);
    State::set_training(true);

    while(true){
        auto batch_size = Compilation::get_batch_size();

        for (int i = 0; i < batch_size; ++i){
            State::reset_trace();
            call_f_default_params(f);
            Compilation::add_trace(State::get_trace());
        }
        Compilation::send_batch();
    }
}


template<class Func, class... Args>
std::vector<std::vector<NDArray<double>>> inference(
        const Func& f,
        const std::tuple<Args...>& args,
        const std::string& tcp_addr,
        size_t n){

    Inference::connect_client(tcp_addr);
    State::set_training(false);

    double sum_w = 0;
    Trace ret;
    for (size_t i = 0; i < n; ++i) {
        State::reset_trace();
        Inference::send_observe_init(detail::to_vec<double>(args));
        call_f_tuple(f, args);
        auto t = State::get_trace();
        auto w = std::exp(t.log_w());
        sum_w += w;
        auto a = w*t;
        ret += a;
    }
    ret /= sum_w;
    return ret.predict();
}
}       // namespace cpprob
#endif //INCLUDE_CPPROB_HPP
