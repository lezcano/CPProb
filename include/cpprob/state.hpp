#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <fstream>
#include <string>
#include <unordered_map>

#include "cpprob/any.hpp"
#include "cpprob/trace.hpp"

namespace cpprob {

enum class StateType {
    compile,
    inference,
    dryrun,
    importance_sampling
};

class State {
public:
    static void reset_trace();

    static TraceCompile get_trace_comp();

    static TracePredicts get_trace_pred();

    static void set(StateType s);

    static StateType current_state();

    static void reset_ids();

    static void serialize_ids_pred(std::ofstream & out_file);

private:
    // Members
    static TraceCompile t_comp;
    static TracePredicts t_pred;
    static StateType state;
    static std::unordered_map<std::string, int> ids_sample;
    static std::unordered_map<std::string, int> ids_predict;
    static Sample prev_sample;
    static Sample curr_sample;



    // Functions so that observe / sample / predict can manipulate the state
    static int register_addr_sample(const std::string& addr);
    static void add_sample(const Sample& s);
    static int sample_instance(int id);

    static void add_observe(const NDArray<double>& x);

    static int register_addr_predict(const std::string& addr);
    static void add_predict(const std::string& addr, const cpprob::any & x);

    static int time_index();
    static void increment_time();

    static void increment_cum_log_prob(double log_p);

    // Friends
    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> &distr, const bool from_observe);

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...>& distr, typename Distr<Params ...>::result_type x);

    friend void predict(const cpprob::any & x, const std::string & addr);
};
}  // namespace cpprob

#endif //CPPROB_STATE_HPP
