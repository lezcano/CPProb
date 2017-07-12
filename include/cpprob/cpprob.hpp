#ifndef INCLUDE_CPPROB_HPP
#define INCLUDE_CPPROB_HPP

#include <array>
#include <cmath>
#include <cstdlib> // std::exit
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cpprob/any.hpp"
#include "cpprob/ndarray.hpp"
#include "cpprob/socket.hpp"
#include "cpprob/state.hpp"
#include "cpprob/trace.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/utils.hpp"
#include "cpprob/distributions/distribution_utils.hpp"

namespace cpprob {
namespace detail {

// Forward declarations
template<class T, class U, class = std::enable_if_t<std::is_arithmetic<U>::value>>
std::vector<T> to_vec(U value);
template<class T, class U, class V>
std::vector<T> to_vec(const std::pair<U, V> & pair);
template<class T, class U>
std::vector<T> to_vec(const std::vector<U> & v);
template<class T, class... Args>
std::vector<T> to_vec(const std::tuple<Args...> & args);
template<class T, class U, std::size_t N>
std::vector<T> to_vec(const std::array<U, N> & args);


template<class T, class U, class>
std::vector<T> to_vec(U value)
{
    return std::vector<T>({static_cast<T>(value)});
}

// Helper function
template<class T, class Iter>
std::vector<T> iter_to_vec(Iter begin, Iter end)
{
    std::vector<T> ret;
    for(; begin != end; ++begin) {
        auto aux = to_vec<T>(*begin);
        ret.insert(ret.end(), aux.begin(), aux.end());

    }
    return ret;
}

template<class T, class U>
std::vector<T> to_vec(const std::vector<U> & v)
{
    return iter_to_vec<T>(v.begin(), v.end());
}

template<class T, class U, std::size_t N>
std::vector<T> to_vec(const std::array<U, N> & arr)
{
    return iter_to_vec<T>(arr.begin(), arr.end());
}

template<class T, class U, class V>
std::vector<T> to_vec(const std::pair<U, V> & pair)
{
    auto ret = to_vec<T>(pair.first);
    auto aux = to_vec<T>(pair.second);

    ret.insert(ret.end(), aux.begin(), aux.end());
    return ret;
}

template<class T, class... Args, size_t... Indices>
std::vector<T> to_vec_tuple(const std::tuple<Args...> & tup, std::index_sequence<Indices...>)
{
    std::vector<T> ret;
    auto append_ret = [&](std::vector<T> && other) { ret.insert(ret.end(), std::make_move_iterator(other.begin()),
                                                                      std::make_move_iterator(other.end())); };
    (void)std::initializer_list<int>{ (append_ret(to_vec<T>(std::get<Indices>(tup))), 0)... };
    return ret;
}

template<class T, class... Args>
std::vector<T> to_vec(const std::tuple<Args...> & args)
{
    return detail::to_vec_tuple<T, Args...>(args, std::make_index_sequence<sizeof...(Args)>());
}
} // end namespace detail

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> & distr, const bool from_observe)
{
    typename Distr<Params ...>::result_type x{};
    std::string addr = get_addr();

    if (State::state() == StateType::compile) {
        x = distr(get_rng());

        if(from_observe){
            StateCompile::add_observe(x);
        }
        else{
            StateCompile::add_sample(Sample{addr, proposal<Distr, Params...>::type_enum, default_distr(distr),
                                     x, StateCompile::sample_instance(addr), StateCompile::time_index()});
        }
        StateCompile::increment_time();

    }
    else if (State::state() == StateType::inference){
        StateInfer::curr_sample_ = Sample(addr, proposal<Distr, Params...>::type_enum, default_distr(distr));

        try {
            auto proposal = SocketInfer::get_proposal<Distr, Params...>(StateInfer::curr_sample_, StateInfer::prev_sample_);

            x = proposal(get_rng());

            StateInfer::increment_log_prob(logpdf(distr, x) - logpdf(proposal, x));
        }
        // No proposal -> Default to prior as proposal
        catch (const std::runtime_error &) {
            x = distr(get_rng());
        }

        StateInfer::curr_sample_.set_value(x);
        StateInfer::prev_sample_ = std::move(StateInfer::curr_sample_);
    }
    else {
        std::cerr << "Incorrect branch in sample_impl!!" << std::endl;
    }

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...> & distr, bool control = false)
{
    if (!control ||
        State::state() == StateType::dryrun ||
        State::state() == StateType::importance_sampling){
        return distr(get_rng());
    }
    else {
        return sample_impl(distr, false);
    }
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...> & distr, const typename Distr<Params ...>::result_type & x)
{
    if (State::compile()){
        sample_impl(distr, true);
    }
    else if (State::inference()) {
        StateInfer::increment_log_prob(logpdf(distr, x));
    }
}

// Declared in state.hpp in class StateCompile with addr=""
template<class T>
void predict(const T & x, const std::string & addr)
{
    if (State::inference()) {
        if (addr.empty()) {
            StateInfer::add_predict(x, get_addr());
        }
        else {
            StateInfer::add_predict(x, addr);
        }
    }
}

template<class Func>
void compile(const Func & f, const std::string & tcp_addr, const std::string & dump_folder, std::size_t batch_size)
{
    State::set(StateType::compile);
    const bool to_file = !dump_folder.empty();

    if (to_file) {
        SocketCompile::config_file(dump_folder);
    }
    else {
        SocketCompile::connect_server(tcp_addr);
    }

    for (std::size_t i = 0; /* Forever */ ; ++i) {
        std::cout << "Generating batch " << i << std::endl;
        if (!to_file) {
            batch_size = SocketCompile::get_batch_size();
        }

        for (std::size_t i = 0; i < batch_size; ++i) {
            StateCompile::new_trace();
            call_f_default_params(f);
            StateCompile::add_trace();
        }
        StateCompile::send_batch();
    }
}

namespace detail {
    // We just support either one tensorial observe or many scalar observes
    template<class... Args, std::enable_if_t<sizeof...(Args) == 1, int> = 0>
    void send_observe_init(const std::tuple<Args...> & observes) {
        SocketInfer::send_observe_init(std::get<0>(observes));
    }

    template<class... Args, std::enable_if_t<sizeof...(Args) != 1, int> = 0>
    void send_observe_init(const std::tuple<Args...> & observes) {
        SocketInfer::send_observe_init(detail::to_vec<double>(observes));
    }
} // end namespace detail

template<class Func, class... Args>
void generate_posterior(
        const Func & f,
        const std::tuple<Args...> & observes,
        const std::string & tcp_addr,
        const std::string & file_name,
        std::size_t n,
        const StateType state)
{
    static_assert(sizeof...(Args) != 0, "The function has to receive the observed values as parameters.");

    State::set(state);
    if (State::state() == StateType::inference) {
        SocketInfer::connect_client(tcp_addr);
    }

    SocketInfer::config_file(file_name);

    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "Generating trace " << i << std::endl;
        StateInfer::new_trace();

        if (State::state() == StateType::inference) {
            // TODO(Lezcano) C++17 This should be done with an if constexpr
            detail::send_observe_init(observes);
        }

        call_f_tuple(f, observes);
        StateInfer::add_trace();
    }
    StateInfer::finish();
}

} // end namespace cpprob
#endif //INCLUDE_CPPROB_HPP
