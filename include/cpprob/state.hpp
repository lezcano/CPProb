#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <cstdint>                                       // for int32_t
#include <iostream>                                     // for size_t
#include <map>                                          // for map
#include <string>                                       // for string
#include <type_traits>                                  // for enable_if_t
#include <utility>                                      // for make_pair
#include <vector>                                       // for vector

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"  // for proposal
#include "cpprob/ndarray.hpp"                           // for NDArray
#include "cpprob/sample.hpp"                            // for Sample
#include "cpprob/socket.hpp"                            // for SocketInfer
#include "cpprob/trace.hpp"                             // for TraceInfer

namespace cpprob {

template<class Distribution>
typename Distribution::result_type sample(Distribution & distr, const bool control = false);
template<class T>
void predict(const T & x, const std::string & addr="");

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
    static flatbuffers::FlatBufferBuilder buff_;

    // Functions so that observe / sample / predict can manipulate the state
    template<class Distr>
    static void add_sample( const std::string & addr,
                            const Distr & distr,
                            const NDArray<double> val)
    {
        auto sample = Sample(addr, distr, val, StateCompile::sample_instance(addr), StateCompile::time_index());

        if (State::rejection_sampling()) {
            StateCompile::trace_.samples_rejection_.emplace_back(std::move(sample));
        }
        else {
            StateCompile::trace_.samples_.emplace_back(std::move(sample));
        }
    }

    static void add_observe(const NDArray<double> & x);

    static int sample_instance(const std::string & addr);

    static int time_index();
    static void increment_time();

    // Functions to handle accept / reject
    static void start_rejection_sampling();
    static void finish_rejection_sampling();

    // Friends
    template<class Distribution>
    friend typename Distribution::result_type sample(Distribution & distr, const bool control);
    template<class Distribution>
    friend void observe(Distribution & distr, const typename Distribution::result_type & x);

    friend class State;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////         Inference             ////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Forward declarations
template<class T, class U, int>
std::vector<T> to_vec(U value);
template<class T, class U, class V>
std::vector<T> to_vec(const std::pair<U, V> & pair);
template<class T, class U>
std::vector<T> to_vec(const std::vector<U> & v);
template<class T, class... Args>
std::vector<T> to_vec(const std::tuple<Args...> & args);
template<class T, class U, std::size_t N>
std::vector<T> to_vec(const std::array<U, N> & args);


template<class T, class U, std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>
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

template<class T, class... Args, std::size_t... Indices>
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


class StateInfer {
public:

    template<class... Args>
    static void send_observe_init(const std::tuple<Args...> & observes)
    {
        NDArray<double> obs_nd = StateInfer::obs_to_ndarr(observes,
                                                          std::integral_constant<bool,
                                                                  std::tuple_size<decltype(
                                                                  discard_build(std::declval<std::tuple<Args...>>())
                                                                  )>::value == 1>{});

        auto observe_init = infcomp::protocol::CreateObservesInitRequest(
                buff_,
                infcomp::protocol::CreateNDArray(buff_,
                                                 buff_.CreateVector<double>(obs_nd.values()),
                                                 buff_.CreateVector<int32_t>(obs_nd.shape())));

        auto msg = infcomp::protocol::CreateMessage(
                buff_,
                infcomp::protocol::MessageBody::ObservesInitRequest,
                observe_init.Union());
        buff_.Finish(msg);
        SocketInfer::send_observe_init(buff_);
        buff_.Clear();
    }

    static void start_infer();
    static void finish_infer();

    static void start_trace();
    static void finish_trace();

private:

    // Attributes
    static TraceInfer trace_;
    static flatbuffers::FlatBufferBuilder buff_;

    static bool all_int_empty;
    static bool all_real_empty;
    static bool all_any_empty;

    // Functions to clear caches of sampling objects
    static std::vector<void (*)()>clear_functions_;

    static void clear_empty_flags();

    static void increment_log_prob(const double log_p);

    template<class Distribution>
    static typename proposal<Distribution>::type get_proposal()
    {
        // Rejection Sampling Cache
        static std::map<std::string, typename proposal<Distribution>::type> cache;

        std::string addr;
        typename decltype(cache)::iterator distr_iter;

        if (State::rejection_sampling()){
            // If it is the first distribution<Distr, Params>, register the clear_function
            if (cache.empty()) {
                // Callback to a static member function
                struct ClearFunction {
                    static void clear () { cache.clear(); }
                };
                clear_functions_.emplace_back(&ClearFunction::clear);
            }

            addr = trace_.curr_sample_.address();
            distr_iter = cache.lower_bound(addr);

            // If it was in the cache, then we return the distribution
            if (distr_iter->first == addr) {
                return distr_iter->second;
            }
        }
        // If the distribution wasn't in the cache or we're not in rejection sampling
        // we ask for the distribution
        auto curr = trace_.curr_sample_.pack(buff_);
        auto last = trace_.prev_sample_.pack(buff_);
        auto msg = infcomp::protocol::CreateMessage(
                buff_,
                infcomp::protocol::MessageBody::ProposalRequest,
                infcomp::protocol::CreateProposalRequest(buff_, curr, last).Union());

        buff_.Finish(msg);
        const auto distr = SocketInfer::get_proposal<Distribution>(buff_);
        buff_.Clear();

        if (State::rejection_sampling()) {
            cache.emplace_hint(distr_iter, std::make_pair(addr, distr));
        }
        return distr;
    }

    // Functions to handle accept / reject
    static void finish_rejection_sampling();

    // Functions to manipulate samples
    template<class Distr>
    static void new_sample( const std::string & addr,
                     const Distr & distr)
    {
        trace_.prev_sample_ = trace_.curr_sample_;
        trace_.curr_sample_ = Sample(addr, distr);
    }

    static void add_value_to_sample(const NDArray<double> & x);

    // TODO(Lezcano) Hack to get around the fact that we do not support many multidimensional observes
    // TODO(Lezcano) C++17 This should be done with an if constexpr
    // We just support either one tensorial observe or many scalar observes
    template<class... Args>
    static NDArray<double> obs_to_ndarr(const std::tuple<Args...> & observes, std::true_type)
    {
        return std::get<0>(observes);
    }

    template<class... Args>
    static NDArray<double> obs_to_ndarr(const std::tuple<Args...> & observes, std::false_type)
    {
        return NDArray<double>(detail::to_vec<double>(discard_build(observes)));
    }


    // TODO(Lezcano) C++17: if constexpr would be nice...
    template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
    static void add_predict(const T & x, const std::string & addr)
    {
        const auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_int_.emplace_back(id, x);
    }

    template<class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
    static void add_predict(const T & x, const std::string & addr)
    {
        const auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T>
    static void add_predict(const NDArray<T> & x, const std::string & addr)
    {
        const auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T, std::enable_if_t<!std::is_integral<T>::value && !std::is_floating_point<T>::value, int> = 0>
    static void add_predict(const T & x, const std::string & addr)
    {
        const auto id = TraceInfer::register_addr_predict(addr);
        trace_.predict_any_.emplace_back(id, x);
    }

    // Friends
    template<class Distribution>
    friend typename Distribution::result_type sample(Distribution & distr, const bool control);
    template<class Distribution>
    friend void observe(Distribution & distr, const typename Distribution::result_type & x);

    template<class T>
    friend void predict(const T & x, const std::string & addr);

    friend class State;
};

}  // namespace cpprob

#endif //CPPROB_STATE_HPP
