#ifndef CPPROB_MODEL_HPP
#define CPPROB_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <iterator>
#include <type_traits>
#include <utility>
#include <map>
#include <vector>

#include "cpprob/utils.hpp" // cpprob::get_zero

namespace cpprob {

template<class T, class WeightType = double>
class EmpiricalDistribution {
public:

    void add_point(const T & value, const WeightType logw)
    {
        x_logw_.emplace_back(std::make_pair(value, logw));
    }

    std::size_t num_points() const
    {
        return x_logw_.size();
    }

    std::map<T, WeightType> distribution() const
    {
        std::map<T, WeightType> ret;

        const auto log_norm = log_normalisation_constant();

        for(const auto & point : x_logw_) {
            ret[point.first] += std::exp(point.second - log_norm);
        }
        return ret;
    }

    T max_a_posteriori() const
    {
        max_a_posteriori(distribution());
    }

    T max_a_posteriori(const std::map<T, WeightType> & distr) const
    {
        return std::max_element(distr.begin(), distr.end(), [](const auto & a, const auto & b) { return a.second < b.second; })->first;
    }

    NDArray<WeightType> raw_moment(const int n) const
    {
        using std::exp;
        if (x_logw_.size() == 0) {
            return NDArray<WeightType>();
        }

        auto log_norm = log_normalisation_constant();

        NDArray<WeightType> ret = get_zero(x_logw_.front().first);
        for(const auto & elem : x_logw_) {
            ret += exp(elem.second - log_norm) * fast_pow(elem.first, n);
        }
        return ret;
    }

    NDArray<WeightType> mean() const
    {
        return raw_moment(1);
    }

    NDArray<WeightType> variance() const
    {
        variance(mean());
    }

    NDArray<WeightType> variance(const NDArray<WeightType> & mean) const
    {
        return raw_moment(2) - mean * mean;
    }

    NDArray<WeightType> std() const
    {
        return sqrt(variance());
    }

    NDArray<WeightType> std(const NDArray<WeightType> & mean) const
    {
        return sqrt(variance(mean));
    }

private:

    template<class Base, class Exp>
    Base fast_pow(Base a, Exp b) const
    {
        static_assert(std::is_integral<Exp>::value, "The type of the exponent is not an integral.");
        if (b == 0) return 1;
        if (b == 1) return a;

        Base aux = a;
        Base result = 1;
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

    WeightType log_normalisation_constant() const
    {

        std::vector<WeightType> log_weights;
        std::transform(x_logw_.begin(), x_logw_.end(), std::back_inserter(log_weights), [](const auto & elem){ return elem.second; });
        return logsumexp(log_weights.begin(), log_weights.end());
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
    std::vector<std::pair<T, WeightType>> x_logw_; // [(value, log weight)]
};
} // end namespace cpprob
#endif //CPPROB_MODEL_HPP
