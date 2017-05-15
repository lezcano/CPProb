#ifndef CPPROB_MODEL_HPP
#define CPPROB_MODEL_HPP

#include <utility>
#include <type_traits>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "cpprob/serialization.hpp"
#include "models/models.hpp"

namespace cpprob {

template<class RealType = double>
class Model {
public:

    template<class CharT, class Traits>
    void load_points(std::basic_istream<CharT, Traits> & is)
    {
        using namespace detail;
        std::pair< // ([(addr, value)], weight) - Weighted Trace
                std::vector< // [(addr, value)] - Trace
                        std::pair<int, NDArray<RealType>> // (addr, value)
                >
                , RealType> val;
        while(is >> val)
            points_.emplace_back(std::move(val));
    }

    template<class CharT, class Traits>
    void load_ids(std::basic_istream<CharT, Traits> & is)
    {
        using namespace detail;
        is >> ids_;
    }

    template<class CharT, class Traits>
    void print_ids(std::basic_ostream<CharT, Traits> & os) const
    {
        using namespace detail;
        for (std::size_t i = 0; i < ids_.size(); ++i) {
            os << i << os.widen(" ") << ids_[i] << std::endl;
        }
    }

#include <iostream>

    RealType raw_moment(const int addr, const int instance, const int n) const
    {
        std::vector<RealType> weights;
        std::transform(points_.begin(), points_.end(), std::back_inserter(weights), [](const auto & elem){ return elem.second; });
        auto log_normalisation_constant = logsumexp(weights.begin(), weights.end());

        std::vector<RealType> w, x;

        for (const auto & trace : points_ ) {
            int count = -1;
            auto trace_logweight = trace.second;
            for (const auto & elem : trace.first) {
                if (elem.first == addr) {
                    ++count;
                    if (count == instance) {
                        //
                        // elem.second.values().front() should be elem.second when NDArray gets the proper overloads
                        w.emplace_back(std::exp(trace_logweight - log_normalisation_constant));
                        x.emplace_back(elem.second.values().front());
                        break;
                    }
                }
            }
        }
        auto it_x = x.begin(), it_w = w.begin();
        RealType ret = 0;
        for(; it_x != x.end() || it_w != w.end(); ++it_w, ++it_x) {
            ret += *it_w * fast_pow(*it_x, n);
        }
        return ret;
    }

    RealType mean(const int addr, const int instance) const
    {
        return raw_moment(addr, instance, 1);
    }

    RealType variance(const int addr, const int instance) const
    {
        auto m = mean(addr, instance);
        return raw_moment(addr, instance, 2) - m * m;
    }

    RealType std(const int addr, const int instance) const
    {
        return std::sqrt(variance(addr, instance));
    }

private:

    template<class T, class U>
    T fast_pow(T a, U b) const
    {
        static_assert(std::is_integral<U>::value, "The type of the exponent is not an integral.");
        if (b == 0) return 1;
        if (b == 1) return a;

        T aux = a;
        T result = 1;
        while (b != 0) {
            if (b % 2 == 0) {
                aux *= aux;
                b /= 2;
            }
            else {
                result *= aux;
                b -= 1;
            }
        }
        return result;
    }

    template<class Iter>
    RealType logsumexp(Iter begin, Iter end) const
    {
        auto max = *std::max_element(begin, end);
        auto exp = std::accumulate(begin, end, 0.0, [max](RealType acc, RealType next) { return acc + std::exp(next-max); });
        return std::log(exp) + max;
    }

    // Attributes
    std::vector< // [([(addr, value)], weight)] - List of Weighted Traces
        std::pair< // ([(addr, value)], weight) - Weighted Trace
            std::vector< // [(addr, value)] - Trace
                std::pair<int, NDArray<RealType>> // (addr, value)
            >
        , RealType>
    > points_;
    std::vector<std::string> ids_;

};

template<class RealType>
void print_stats_model(const Model<RealType> & m, const decltype(&cpprob::models::gaussian_unknown_mean)& f) {
    if (f == cpprob::models::gaussian_unknown_mean) {
        std::cout << "Mean : " << m.mean(0, 0) << std::endl;
        std::cout << "Sigma : " << m.std(0, 0) << std::endl;
    }
}

template<class RealType, std::size_t N>
void print_stats_model(const Model<RealType> & m, void (*f)(const std::array<RealType, N>&)) {
    if (f == &cpprob::models::hmm<N>) {
        for (std::size_t i = 0; i < N; ++i) {
            std::cout << "Mean " << i << ": " << m.mean(0, i) << std::endl;
        }
    }
    else if (f == &cpprob::models::linear_gaussian_1d<N>) {
        for (std::size_t i = 0; i < N; ++i) {
            std::cout << "Mean " << i << ": " << m.mean(0, i) << std::endl;
        }
    }
}

} // end namespace cpprob

#endif //CPPROB_MODEL_HPP
