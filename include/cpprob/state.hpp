#ifndef CPPROB_STATE_HPP
#define CPPROB_STATE_HPP

#include <cstdint>                                       // for int32_t
#include <iostream>                                     // for size_t
#include <map>                                          // for map
#include <string>                                       // for string
#include <type_traits>                                  // for enable_if_t
#include <utility>                                      // for make_pair
#include <vector>                                       // for vector
#include <boost/any.hpp>                                // for any

#include <boost/filesystem/path.hpp>                    // for path

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"          // for proposal
#include "cpprob/ndarray.hpp"                           // for NDArray
#include "cpprob/sample.hpp"                            // for Sample
#include "cpprob/socket.hpp"                            // for SocketInfer
#include "cpprob/trace.hpp"                             // for TraceInfer

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
//////////////////////////          State             //////////////////////////
////////////////////////////////////////////////////////////////////////////////

enum class StateType {
    compile,
    csis,
    sis,
    dryrun
};

class State {
public:
    // Accept / Reject Sampling
    static void start_rejection_sampling();
    static void finish_rejection_sampling();

    // State set
    static void set(StateType s);

    // Query
    static bool rejection_sampling();
    static bool compile ();
    static bool csis ();
    static bool sis ();
    static bool dryrun ();

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
                            Distr && distr,
                            const typename std::decay_t<Distr>::result_type & val)
    {
        auto sample = Sample(addr, std::forward<Distr>(distr), val);

        if (State::rejection_sampling()) {
            StateCompile::trace_.samples_rejection_.emplace_back(std::move(sample));
        }
        else {
            StateCompile::trace_.samples_.emplace_back(std::move(sample));
        }
    }

    static void add_observe(const NDArray<double> & x);
    static void add_observe(NDArray<double> && x);

    // Functions to handle accept / reject
    static void finish_rejection_sampling();

    // Friends
    template<class Distribution, class String>
    friend auto sample(Distribution && distr, const bool control, String && address);
    template<class Distribution>
    friend void observe(Distribution && distr, const typename std::decay_t<Distribution>::result_type & x);
    template<class T>
    friend void metaobserve(T && x);

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
    static void send_start_inference(const std::tuple<Args...> & observes)
    {
        NDArray<double> obs_nd = StateInfer::obs_to_ndarray(observes,
                                                            std::integral_constant<bool, sizeof...(Args) == 1>{});

        auto observe_init = protocol::CreateRequestStartInference(
                buff_,
                protocol::CreateNDArray(buff_,
                                        buff_.CreateVector<double>(obs_nd.values()),
                                        buff_.CreateVector<int32_t>(obs_nd.shape_int())));

        auto msg = protocol::CreateMessage(
                buff_,
                protocol::MessageBody::RequestStartInference,
                observe_init.Union());
        buff_.Finish(msg);
        SocketInfer::send_start_inference(buff_);
        buff_.Clear();
    }

    static void config_file(const boost::filesystem::path & dump_file);

    static void start_infer();
    static void finish_infer();

    static void start_trace();
    static void finish_trace();

private:

    // Attributes
    static TraceInfer trace_;
    static flatbuffers::FlatBufferBuilder buff_;
    static std::map<std::string, double> log_prob_rej_samp_;
    static boost::filesystem::path dump_file_;

    static bool all_int_empty;
    static bool all_real_empty;
    static bool all_any_empty;

    // Functions to clear caches of sampling objects
    static std::vector<void (*)()>clear_functions_;

    static void clear_empty_flags();

    static void increment_log_prob(const double log_p, const std::string & addr="");

    template<class Proposal>
    static Proposal get_proposal()
    {
        // Rejection Sampling Cache
        static std::map<std::string, Proposal> cache;

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
        auto msg = protocol::CreateMessage(
                buff_,
                protocol::MessageBody::RequestProposal,
                protocol::CreateRequestProposal(buff_, curr, last).Union());

        buff_.Finish(msg);
        const auto distr = SocketInfer::get_proposal<Proposal>(buff_);
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
    static void new_sample( const std::string & addr, const Distr & distr)
    {
        trace_.prev_sample_ = trace_.curr_sample_;
        trace_.curr_sample_ = Sample(addr, distr);
    }

    static void add_value_to_sample(const boost::any & x);

    // TODO(Lezcano) Hack to get around the fact that we do not support many multidimensional observes
    // TODO(Lezcano) C++17 This should be done with an if constexpr
    // We just support either one tensorial observe or many scalar observes
    template<class... Args>
    static NDArray<double> obs_to_ndarray(const std::tuple<Args...> & observes, std::true_type)
    {
        return std::get<0>(observes);
    }

    template<class... Args>
    static NDArray<double> obs_to_ndarray(const std::tuple<Args...> & observes, std::false_type)
    {
        return NDArray<double>(detail::to_vec<double>(observes));
    }


    // TODO(Lezcano) C++17: if constexpr would be nice...
    template<class T, class String,
            std::enable_if_t<std::is_integral<T>::value, int> = 0>
    static void add_predict(T x, String && addr)
    {
        const auto id = TraceInfer::register_addr_predict(std::forward<String>(addr));
        trace_.predict_int_.emplace_back(id, x);
    }

    template<class T, class String,
            std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
    static void add_predict(T x, String && addr)
    {
        const auto id = TraceInfer::register_addr_predict(std::forward<String>(addr));
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T, class String>
    static void add_predict(const NDArray<T> & x, String && addr)
    {
        const auto id = TraceInfer::register_addr_predict(std::forward<String>(addr));
        trace_.predict_real_.emplace_back(id, x);
    }

    template<class T, class String>
    static void add_predict(NDArray<T> && x, String && addr)
    {
        const auto id = TraceInfer::register_addr_predict(std::forward<String>(addr));
        trace_.predict_real_.emplace_back(std::move(id), std::move(x));
    }

    template<class T, class String,
            std::enable_if_t<!std::is_integral<std::decay_t<T>>::value &&
                             !std::is_floating_point<std::decay_t<T>>::value, int> = 0>
    static void add_predict(T && x, String && addr)
    {
        const auto id = TraceInfer::register_addr_predict(std::forward<String>(addr));
        trace_.predict_any_.emplace_back(std::move(id), std::forward<T>(x));
    }

    static void dump_predicts(const std::vector<std::pair<std::size_t, cpprob::any>> & predicts, const double log_w, const boost::filesystem::path & path);
    static void dump_ids(const std::unordered_map<std::string, std::size_t> & ids_predict, const boost::filesystem::path & path);
    static boost::filesystem::path get_file_name(const std::string & value);

    // Friends
    template<class Distribution, class String>
    friend auto sample(Distribution && distr, const bool control, String && address);
    template<class Distribution>
    friend void observe(Distribution && distr, const typename std::decay_t<Distribution>::result_type & x);
    template<class T,  class String>
    friend void predict(T && x, String && addr);
    template<class T>
    friend void predict(T && x);


    friend class State;
};

}  // namespace cpprob

#endif //CPPROB_STATE_HPP
