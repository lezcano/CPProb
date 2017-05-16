#ifndef INCLUDE_CPPROB_HPP
#define INCLUDE_CPPROB_HPP

#include <array>
#include <cmath>
#include <cstdlib> // std::exit
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem/operations.hpp>

#include "cpprob/utils.hpp"
#include "cpprob/trace.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/state.hpp"
#include "cpprob/socket.hpp"
#include "cpprob/ndarray.hpp"

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


template<class T, class U, class = std::enable_if_t<std::is_arithmetic<U>::value>>
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
std::vector<T> to_vec(const std::tuple<Args...> & args){
    return detail::to_vec_tuple<T, Args...>(args, std::make_index_sequence<sizeof...(Args)>());
}
} // end namespace detail

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> & distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = State::register_addr_sample(addr);

    if (State::current_state() == StateType::compile) {
        x = distr(get_rng());

        if(from_observe){
            State::add_observe(x);
        }
        else{
            State::add_sample(Sample{addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum,
                                     default_distr(distr), State::time_index(), x});
        }
    }
    else if  (State::current_state() == StateType::importance_sampling) {
        x = distr(get_rng());
        State::increment_cum_log_prob(logpdf(distr, x));
    }
    else {
        State::curr_sample = Sample(addr, State::sample_instance(id), proposal<Distr, Params...>::type_enum, default_distr(distr));

        try {
            auto proposal = Inference::get_proposal<Distr, Params...>(State::curr_sample, State::prev_sample);

            x = proposal(get_rng());

            State::increment_cum_log_prob(logpdf(distr, x) - logpdf(proposal, x));
        }
        catch (const std::runtime_error &) {
            x = distr(get_rng());
            // We do not increment the log_probability of the trace since p(x)/p(x) = 1
        }

        State::curr_sample.set_value(x);
        State::prev_sample = std::move(State::curr_sample);
    }

    State::increment_time();

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...> & distr, bool control = false) {
    if (!control || State::current_state() == StateType::dryrun){
        return distr(get_rng());
    }
    else {
        return sample_impl(distr, false);
    }
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...> & distr, typename Distr<Params ...>::result_type x) {
    if (State::current_state() == StateType::compile){
        sample_impl(distr, true);
    }
    else if (State::current_state() == StateType::inference ||
             State::current_state() == StateType::importance_sampling){
        State::increment_cum_log_prob(logpdf(distr, x));
    }
}

void predict(const NDArray<double> &x);

void set_state(StateType s);

template<class Func>
void compile(const Func & f, const std::string & tcp_addr, const std::string & dump_folder, const int n) {
    set_state(StateType::compile);
    bool to_file = !dump_folder.empty();

    if (to_file) {
        const boost::filesystem::path path_dump_folder (dump_folder);
        if (!boost::filesystem::exists(path_dump_folder)) {
            std::cerr << "Provided --dump_folder \"" + dump_folder + "\" does not exist.\n"
                      << "Please provide a valid folder.\n";
            std::exit (EXIT_FAILURE);
        }
        Compilation::config_to_file(dump_folder, n);
    }
    else {
        Compilation::connect_server(tcp_addr);
    }

    while(true){
        auto batch_size = Compilation::get_batch_size();

        for (int i = 0; i < batch_size; ++i){
            State::reset_trace();

            // TODO(Lezcano) Hack
            #ifdef BUILD_SHERPA
            f(std::vector<std::vector<std::vector<double>>>());
            #else
            call_f_default_params(f);
            #endif
            Compilation::add_trace(State::get_trace_comp());
        }
        Compilation::send_batch();
    }
}

// TODO (Lezcano) Solve this
// Almost copied from generate_posterior() :(
template<class Func, class... Args>
void importance_sampling(
        const Func & f,
        const std::tuple<Args...> & observes,
        const std::string & file_name,
        size_t n){
    set_state(StateType::importance_sampling);

    std::ofstream out_file(file_name);
    out_file.precision(std::numeric_limits<double>::digits10);

    //double sum_w = 0;
    for (size_t i = 0; i < n; ++i) {
        State::reset_trace();

        // TODO(Lezcano) Hack
        #ifdef BUILD_SHERPA
        f(std::get<0>(observes));
        #else
        call_f_tuple(f, observes);
        #endif

        auto t = State::get_trace_pred();
        out_file << std::scientific << t << '\n';
    }
    std::ofstream ids_file(file_name + "_ids");
    State::serialize_ids_pred(ids_file);
}

template<class Func, class... Args>
void generate_posterior(
        const Func & f,
        const std::tuple<Args...> & observes,
        const std::string & tcp_addr,
        const std::string & file_name,
        size_t n){
    static_assert(sizeof...(Args) != 0, "The function has to receive the observed values as parameters.");

    set_state(StateType::inference);
    Inference::connect_client(tcp_addr);

    std::ofstream out_file(file_name);
    out_file.precision(std::numeric_limits<double>::digits10);

    //double sum_w = 0;
    for (size_t i = 0; i < n; ++i) {
        State::reset_trace();
        // We just support either one tensorial observe or many scalar observes
        if (sizeof...(Args) == 1) {
            Inference::send_observe_init(NDArray<>(std::get<0>(observes)));
        }
        else {
            Inference::send_observe_init(detail::to_vec<double>(observes));
        }

        // TODO(Lezcano) Hack
        #ifdef BUILD_SHERPA
        f(std::get<0>(observes));
        #else
        call_f_tuple(f, observes);
        #endif

        auto t = State::get_trace_pred();
        out_file << std::scientific << t << '\n';
    }
    std::ofstream ids_file(file_name + "_ids");
    State::serialize_ids_pred(ids_file);
}
}       // namespace cpprob
#endif //INCLUDE_CPPROB_HPP
