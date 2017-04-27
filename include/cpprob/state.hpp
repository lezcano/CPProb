#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <unordered_map>

#include "cpprob/trace.hpp"

namespace cpprob {

class State {
public:
    static void reset_trace();

    static Trace get_trace();

    static void set_training(const bool t);

    static void reset_ids();

private:
    static Trace t;
    static bool training;
    static std::unordered_map<std::string, int> ids;
    static Sample prev_sample;
    static Sample curr_sample;

    static int sample_instance(int id);

    static int register_addr(const std::string& addr);

    static void add_sample_to_batch(const Sample& s);
    static void add_observe_to_batch(const NDArray<double>& n);

    static void add_sample_to_trace(const NDArray<double>& x, int id);
    static void add_observe_to_trace(double prob);

    static int time_index();
    static void increment_time();

    static void increment_cum_log_prob(double log_p);

    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> &distr, const bool from_observe);

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...>& distr, typename Distr<Params ...>::result_type x);
};
}  // namespace cpprob

#endif //CPPROB_STATE_HPP
