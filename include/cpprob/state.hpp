#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpprob/any.hpp"
#include "cpprob/sample.hpp"
#include "cpprob/socket.hpp"
#include "cpprob/trace.hpp"
#include "cpprob/distributions/distribution_utils.hpp"

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

class State {
public:
    // Accept / Reject Sampling
    static void start_rejection_sampling();

    static void finish_rejection_sampling();

    static bool rejection_sampling();

    // State set / query
    static void set(StateType s);

    static bool compile ();

    static bool inference ();

    static StateType state();

private:
    static StateType state_;
    static bool rejection_sampling_;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class StateCompile {
public:
    static std::size_t get_batch_size();
    static void start_trace();
    static void finish_trace();

    static void start_batch();
    static void finish_batch();

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

    // Functions to handle accept / reject
    static void start_rejection_sampling();
    static void finish_rejection_sampling();

    // Friends
    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type sample_impl(Distr<Params ...> & distr, const bool from_observe);

    friend class State;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////         Inference             ////////////////////////
////////////////////////////////////////////////////////////////////////////////


class StateInfer {
public:

    static void start_infer();
    static void finish_infer();

    static void start_trace();
    static void finish_trace();

private:

    template<template <class ...> class Distr, class ...Params>
    class DistributionCache {
    public:

       static void emplace_hint(typename std::map<std::string, typename proposal<Distr, Params...>::type>::const_iterator hint,
                                const std::string & addr,
                                const typename proposal<Distr, Params...>::type & distr)
        {
            distributions_.emplace_hint(hint, std::make_pair(addr, distr));
        }

        static auto distribution(const std::string & addr)
        {
            return distributions_.lower_bound(addr);
        }

        static bool first_distribution()
        {
            return distributions_.empty();
        }

        static void clear()
        {
            distributions_.clear();
        }

    private:
        static std::map<std::string, typename proposal<Distr, Params...>::type> distributions_;
    };

    // Attributes
    static TraceInfer trace_;
    static Sample prev_sample_;
    static Sample curr_sample_;

    static bool all_int_empty;
    static bool all_real_empty;
    static bool all_any_empty;

    // Functions to clear caches of sampling objects
    static std::vector<void (*)()>clear_functions_;

    static void clear_empty_flags();

    static void increment_log_prob(const double log_p);

    template<template <class ...> class Distr, class ...Params>
    static typename proposal<Distr, Params ...>::type get_proposal()
    {
        if (!State::rejection_sampling()) {
            return SocketInfer::get_proposal<Distr, Params...>(curr_sample_, prev_sample_);
        }
        else {
            // If it is the first distribution<Distr, Params>, register the clear_function
            if (DistributionCache<Distr, Params...>::first_distribution()) {
                clear_functions_.emplace_back(&DistributionCache<Distr, Params...>::clear);
            }

            auto addr = curr_sample_.sample_address();
            auto distr_iter = DistributionCache<Distr, Params...>::distribution(addr);

            if (distr_iter->first == addr) {
                return distr_iter->second;
            }
            else {
                auto distr = SocketInfer::get_proposal<Distr, Params...>(curr_sample_, prev_sample_);
                DistributionCache<Distr, Params...>::emplace_hint(distr_iter, addr, distr);
                return distr;
            }
        }
    }

    // Functions to handle accept / reject
    static void finish_rejection_sampling();

    // if constexpr would be nice...
    template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    static void add_predict(const T & x, const std::string & addr)
    {
        auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_int_.emplace_back(id, x);
    }

    template<class T, std::enable_if_t<!std::is_integral<T>::value &&
                                        std::is_constructible<NDArray<double>, T>::value
                                        , int> = 0>
    static void add_predict(const T & x, const std::string & addr)
    {
        auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T, std::enable_if_t<!std::is_integral<T>::value &&
                                       !std::is_constructible<NDArray<double>, T>::value
                                       , int> = 0>
    static void add_predict(const T & x, const std::string & addr)
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
    friend void predict(const T & x, const std::string & addr="");

    friend class State;
};

template<template <class ...> class Distr, class ...Params>
std::map<std::string, typename proposal<Distr, Params...>::type> StateInfer::DistributionCache<Distr, Params...>::distributions_{};
}  // namespace cpprob

#endif //CPPROB_STATE_HPP
