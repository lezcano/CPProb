#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <unordered_map>
#include <string>

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
    static std::unordered_map<std::string, int> ids_sample;
    static std::unordered_map<std::string, int> ids_predict;
    static Sample prev_sample;
    static Sample curr_sample;


    static int register_addr_sample(const std::string& addr);
    static void add_sample(const Sample& s);
    static int sample_instance(int id);

    static void add_observe(const NDArray<double>& x);

    static int register_addr_predict(const std::string& addr);
    static void add_predict(const std::string& addr, const NDArray<double>& x);

    static int time_index();
    static void increment_time();

    static void increment_cum_log_prob(double log_p);

    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> &distr, const bool from_observe);

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...>& distr, typename Distr<Params ...>::result_type x);

    friend void predict(const NDArray<double>& x);
};
}  // namespace cpprob

#endif //CPPROB_STATE_HPP
