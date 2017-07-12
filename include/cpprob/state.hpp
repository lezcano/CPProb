#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <fstream>
#include <string>
#include <unordered_map>

#include "cpprob/any.hpp"
#include "cpprob/sample.hpp"
#include "cpprob/trace.hpp"

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
//////////////////////////          State             //////////////////////////
////////////////////////////////////////////////////////////////////////////////

enum class StateType {
    compile,
    inference,
    dryrun,
    importance_sampling
};

class StateCompile;
class StateInfer;


class State {
public:
    static void set(StateType s);

    static bool compile ();

    static bool inference ();

    static StateType state();

    static void new_model();
private:
    static StateType state_;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class TraceCompile;

template<class T>
void predict(const T & x, const std::string & addr="");

class StateCompile {
public:
    static std::size_t get_batch_size();
    static void add_trace();
    static void send_batch();

    static void new_trace();
    static void new_model();

private:
    // Attributes
    static TraceCompile trace_;
    static std::vector<TraceCompile> batch_;

    // Functions so that observe / sample / predict can manipulate the state
    static void add_sample(const Sample & s);
    static int sample_instance(const std::string & addr);

    static int time_index();
    static void increment_time();

    static void add_observe(const NDArray<double> & x);

    // Friends
    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> & distr, const bool from_observe);

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...> & distr, typename Distr<Params ...>::result_type x);

    template<class T>
    friend void predict(const T & x, const std::string & addr);
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////         Inference             ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class StateInfer {
public:
    static void add_trace();
    static void finish();

    static void new_trace();
    static void new_model();

private:
    // Attributes
    static TraceInfer trace_;
    static Sample prev_sample_;
    static Sample curr_sample_;

    static bool all_int_empty;
    static bool all_real_empty;
    static bool all_any_empty;
    
    static void clear_empty_flags();

    static void increment_log_prob(const double log_p);

    template<class T>
    static std::enable_if_t<std::is_integral<T>::value, void>
    add_predict(const T & x, const std::string & addr)
    {
        auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_int_.emplace_back(id, x);
    }

    template<class T>
    static std::enable_if_t<std::is_floating_point<T>::value, void>
    add_predict(const T & x, const std::string & addr)
    {
        auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T>
    static std::enable_if_t<!std::is_integral<T>::value && !std::is_floating_point<T>::value, void>
    add_predict(const T & x, const std::string & addr)
    {
        auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_any_.emplace_back(id, x);
    }

    // Friends
    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> & distr, const bool from_observe);

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...> & distr, const typename Distr<Params ...>::result_type & x);

    template<class T>
    friend void predict(const T & x, const std::string & addr);
};
}  // namespace cpprob

#endif //CPPROB_STATE_HPP
