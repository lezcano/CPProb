#ifndef INCLUDE_CPPROB_HPP_
#define INCLUDE_CPPROB_HPP_

#include <execinfo.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <cmath>
#include <array>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <utility>

#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "traits.hpp"

namespace cpprob {

// Forward declaration to declare the function as a friend of Core
template<class Func>
std::vector<std::vector<double>>
expectation(const Func& f,
            size_t n = 100000,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q
                = [](std::vector<std::vector<double>> x) {return x;});

template<bool Testing>
class Core{
public:

    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample(const Distr<Params ...>& distr) {
        return sample_impl(distr, false);
    }

    template<template <class ...> class Distr, class ...Params>
    void observe(const Distr<Params ...>& distr, double x) {
        if (Testing) {
            sample_impl(distr, true);
            return;
        }
        using std::log;
        auto prob = pdf(math_distr(distr), x);
        w_ += log(prob);
        y_.emplace_back(prob);
    }

private:


    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample_impl(const Distr<Params ...>& distr, bool from_observe) {
        std::string addr = get_addr(from_observe);

        // TODO(Lezcano) Not parallelizable right now, ids_ is static
        auto id = Core::ids_.emplace(addr, static_cast<int>(Core::ids_.size())).first->second;

        typename Distr<Params ...>::result_type x;
        if (Testing) {
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<boost::random::mt19937, Distr<Params ...>> next_val{rng, distr};
            x = next_val();
        } else {
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<
                boost::random::mt19937,
                Distr<Params ...>>
                    next_val{rng, distr};
            // TODO(Lezcano) Use last_x_ and id to compute x
            x = next_val();
            last_x_ = x;
        }


        if (id >= x_.size())
            x_.resize(id + 1);

        x_[id].emplace_back(static_cast<double>(x));

        x_addr_.emplace_back(static_cast<double>(x), id);

        return x;
    }



    template<class Func>
    friend std::vector<std::vector<double>>
    expectation(const Func&,
                size_t,
                const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>&);

    template<typename Func>
    friend Core<false> eval(Func f);


    std::string get_addr(bool from_observe) const {
        constexpr int buf_size = 1000;
        static void *buffer[buf_size];
        char **strings;

        size_t nptrs = backtrace(buffer, buf_size);

        // We will not store the call to get_traces or the call to sample
        // If we came from an observe statement we discard 3 (observe -> sample -> get_addr)
        const size_t str_discarded = from_observe ? 3 : 2;
        std::vector<std::string> trace;

        strings = backtrace_symbols(buffer, nptrs);
        if (strings == nullptr) {
            perror("backtrace_symbols");
            exit(EXIT_FAILURE);
        }

        // The -2 is to discard the call to _start and
        // the call to __libc_start_main
        for (size_t j = str_discarded; j < nptrs - 2; j++) {
            std::string s {strings[j]};

            // The +3 is to discard the characters
            auto first = s.find("[0x") + 3;
            auto last = s.find("]");
            trace.emplace_back(s.substr(first, last-first));
        }
        free(strings);
        return std::accumulate(trace.begin(), trace.end(), std::string(""));
    }

    std::vector<std::pair<double, int>> x_addr() const {
        return x_addr_;
    }

    std::vector<double> y() const {
        return y_;
    }

    // Idea from
    // http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
    template<class T = boost::random::mt19937, std::size_t N = T::state_size>
    std::enable_if_t<N != 0, T> seeded_rng() const {
        std::array<typename T::result_type, N> random_data;
        std::random_device rd;
        std::generate(random_data.begin(), random_data.end(), std::ref(rd));
        std::seed_seq seeds(random_data.begin(), random_data.end());
        return T{seeds};
    }

    double w_ = 0;
    std::vector<std::vector<double>> x_;
    static std::unordered_map<std::string, int> ids_;

    std::vector<std::pair<double, int>> x_addr_;
    std::vector<double> y_;

    double last_x_ = 0;
};

template<bool Inference>
std::unordered_map<std::string, int> cpprob::Core<Inference>::ids_;

template<typename Func>
Core<false> eval(Func f) {
    Core<false> c;
    f(c);
    c.w_ = std::exp(c.w_);
    return c;
}

template<class Func>
std::pair<std::vector<std::pair<double, int>>, std::vector<double>>
train(const Func& f, int n = 10000) {
    Core<true> c;
    for (int i = 0; i < n; ++i)
        f(c);
    // TODO(Lezcano) This can be optimized
    return {c.x_addr(), c.y()};
}


// Default parameters declared in forward declaration
template<class Func>
std::vector<std::vector<double>>
expectation(const Func& f,
            size_t n,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q) {
    Core<false> core;
    double sum_w = 0;
    std::vector<std::vector<double>> ret, aux;

    Core<false>::ids_.clear();

    for (size_t i = 0; i < n; ++i) {
        core = eval(f);
        sum_w += core.w_;
        aux = q(core.x_);

        if (aux.size() > ret.size())
            ret.resize(aux.size());

        // Add new trace weighted with its weight
        for (size_t i = 0; i < aux.size(); ++i) {
            if (aux[i].empty()) continue;
            // Multiply each element sampled x_i of the trace by the weight of the trace
            std::transform(aux[i].begin(),
                           aux[i].end(),
                           aux[i].begin(),
                           [&](double a){ return core.w_ * a; });
            // Put in ret[i] the biggest of the two vectors
            if (aux[i].size() > ret[i].size())
                std::swap(aux[i], ret[i]);

            // Add the vectors
            std::transform(aux[i].begin(),
                           aux[i].end(),
                           ret[i].begin(),
                           ret[i].begin(),
                           std::plus<double>());
        }
    }

    // Normalise (Compute E_\pi)
    for (auto& elem : ret)
        std::transform(elem.begin(),
                       elem.end(),
                       elem.begin(),
                       [sum_w](double e){ return e/sum_w; });

    return ret;
}
}  // namespace cpprob
#endif  // INCLUDE_CPPROB_HPP_
