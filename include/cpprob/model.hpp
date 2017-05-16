#ifndef CPPROB_MODEL_HPP
#define CPPROB_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpprob/serialization.hpp"
#include "cpprob/utils.hpp"
#include "models/models.hpp"
#include "models/sherpa.hpp"
#include "models/sherpa_mini.hpp"

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

    NDArray<RealType> raw_moment(const int addr, const int instance, const int n) const
    {
        using std::log;
        std::vector<RealType> weights;
        std::transform(points_.begin(), points_.end(), std::back_inserter(weights), [](const auto & elem){ return elem.second; });
        auto log_normalisation_constant = logsumexp(weights.begin(), weights.end());

        std::vector<RealType> w;
        std::vector<NDArray<RealType>> x;


        for (const auto & trace : points_ ) {
            int count = -1;
            auto trace_logweight = trace.second;
            for (const auto & elem : trace.first) {
                if (elem.first == addr) {
                    ++count;
                    if (count == instance) {
                        w.emplace_back(std::exp(trace_logweight - log_normalisation_constant));
                        x.emplace_back(elem.second);
                        break;
                    }
                }
            }
        }

        if (x.size() == 0) {
            return NDArray<RealType>();
        }

        auto it_x = x.begin(), it_w = w.begin();
        NDArray<RealType> ret = get_zero(*it_x);
        for(; it_x != x.end() || it_w != w.end(); ++it_w, ++it_x) {
            ret += *it_w * fast_pow(*it_x, n);
        }
        return ret;
    }

    NDArray<RealType> mean(const int addr, const int instance) const
    {
        return raw_moment(addr, instance, 1);
    }

    NDArray<RealType> variance(const int addr, const int instance) const
    {
        auto m = mean(addr, instance);
        return raw_moment(addr, instance, 2) - m * m;
    }

    NDArray<RealType> std(const int addr, const int instance) const
    {
        return sqrt(variance(addr, instance));
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

    typename std::iterator_traits<Iter>::value_type
    logsumexp(Iter begin, Iter end) const
    {
        using cpprob::detail::supremum; // Max for comparable types
        using cpprob::detail::get_zero; // Get zero for arithmetic types
        using std::exp;
        using std::log;
        if (begin == end) {
            return typename std::iterator_traits<Iter>::value_type();
        }

        auto max = supremum(begin, end);
        auto exp_val = std::accumulate(begin, end,
                                   get_zero(*begin),
                                   [&max](const auto & acc,
                                              const auto & next){ return acc + exp(next-max); });
        return log(exp_val) + max;
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
void print_stats_model(const Model<RealType> & m, decltype(&cpprob::models::gaussian_unknown_mean) f) {
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

template<class RealType>
void print_stats_model(const Model<RealType> & m, decltype(&cpprob::models::sherpa_mini_wrapper) f) {
    if (f == cpprob::models::sherpa_mini_wrapper) {
        std::cout << "Mean: " << m.mean(0, 0) << std::endl;
    }
}

template<class RealType>
void print_stats_model(const Model<RealType> & m, const cpprob::models::SherpaWrapper &) {
    std::cout << "Mean: " << m.mean(0, 0) << std::endl;
}

} // end namespace cpprob

#endif //CPPROB_MODEL_HPP
