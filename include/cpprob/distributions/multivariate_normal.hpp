#ifndef CPPROB_MULTIVARIATE_NORMAL_HPP_HPP
#define CPPROB_MULTIVARIATE_NORMAL_HPP_HPP

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <tuple> // std::tie
#include <vector>

#include <boost/random/normal_distribution.hpp>

#include "cpprob/serialization.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {

template<typename RealType = double>
class multivariate_normal_distribution {
public:
    // types
    using input_type =  std::vector<RealType>;
    using result_type =  NDArray<RealType>;

    // member classes/structs/unions

    class param_type {
    public:

        // types
        using distribution_type = multivariate_normal_distribution;

        // construct/copy/destruct
        param_type() : distr_{boost::random::normal_distribution<RealType>(0, 1)}, shape_{1} {}

        template<class Iter>
        param_type(Iter mean_first, Iter mean_last, RealType covariance)
        {
            init(mean_first, mean_last, covariance);
            init_shape();
        }

        template<class IterMean, class IterSigma>
        param_type(IterMean mean_first, IterMean mean_last, IterSigma covariance_first, IterSigma covariance_last)
        {
            init(mean_first, mean_last, covariance_first, covariance_last);
            init_shape();
        }

        param_type(const std::initializer_list<RealType> & mean, RealType covariance)
        {
            init(mean.begin(), mean.end(),covariance);
            init_shape();
        }

        param_type(const std::initializer_list<RealType> & mean,  const std::initializer_list<RealType> & covariance)
        {
            init(mean.begin(), mean.end(), covariance.begin(), covariance.end());
            init_shape();
        }

        template<typename RangeMean>
        explicit param_type(const RangeMean & mean,  RealType covariance)
        {
            init(mean.begin(), mean.end(), covariance);
            init_shape();
        }

        template<typename RangeMean, class RangeSigma>
        explicit param_type(const RangeMean & mean,  const RangeSigma & covariance)
        {
            init(mean.begin(), mean.end(), covariance.begin(), covariance.end());
            init_shape();
        }

        param_type(const NDArray<RealType> & mean,  RealType covariance)
        {
            auto mean_v = mean.values();
            init(mean_v.begin(), mean_v.end(), covariance);
            init_shape(mean.shape());
        }

        template<class IterSigma>
        param_type(const NDArray<RealType> & mean, IterSigma covariance_first, IterSigma covariance_last)
        {
            auto mean_v = mean.values();
            init(mean_v.begin(), mean_v.end(), covariance_first, covariance_last);
            init_shape(mean.shape());
        }

        // public member functions
        NDArray<RealType> mean() const
        {
            std::vector<RealType> mean;
            std::transform(distr_.begin(), distr_.end(), std::back_inserter(mean), [](const auto & distr) { return distr.mean(); });
            return NDArray<RealType>(std::move(mean), shape_);
        }

        std::vector<RealType> covariance() const
        {
            std::vector<RealType> covariance;
            std::transform(distr_.begin(), distr_.end(), std::back_inserter(covariance), [](const auto & distr) { return distr.sigma()*distr.sigma(); });
            return covariance;
        }

        std::vector<std::size_t> shape() const
        {
            return shape_;
        }

        std::vector<boost::random::normal_distribution<RealType>> distr() const
        {
            return distr_;
        }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream< CharT, Traits > &
        operator<<(std::basic_ostream< CharT, Traits > & os, const param_type & param)
        {
            using namespace detail; // operator<< for std::vector
            return os << param.mean() << os.widen(' ') << param.covariance();
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream< CharT, Traits > &
        operator>>(std::basic_istream< CharT, Traits > & is, param_type &  param)
        {
            using namespace detail; // operator>> for std::vector
            std::vector<RealType> mean_temp, covariance_temp;
            if(!(is >> mean_temp >> std::ws >> covariance_temp)) {
                return is;
            }

            if (covariance_temp.size() == 1) {
                param.init(mean_temp.begin(), mean_temp.end(), covariance_temp.front());
                return is;
            }

            param.init(mean_temp.begin(), mean_temp.end(), covariance_temp.begin(), covariance_temp.end());

            return is;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return lhs.distr_ == rhs.distr_;

        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }

    private:

        template<class Iter>
        void init(Iter mean_first, Iter mean_last, RealType covariance)
        {
            using std::sqrt;
            distr_.clear();
            for (; mean_first != mean_last; ++mean_first){
                // boost::random::normal uses the standard deviation
                distr_.emplace_back(*mean_first, sqrt(covariance));
            }
        }

        template<class IterMean, class IterSigma>
        void init(IterMean mean_first, IterMean mean_last, IterSigma covariance_first, IterSigma covariance_last)
        {
            using std::sqrt;
            for (; mean_first != mean_last && covariance_first != covariance_last; ++mean_first, ++covariance_first) {
                // boost::random::normal uses the standard deviation
                distr_.emplace_back(*mean_first, sqrt(*covariance_first));
            }
        }

        void init_shape(std::vector<std::size_t> shape = std::vector<std::size_t>())
        {
            if (shape.empty())
                shape_ = std::vector<std::size_t>{distr_.size()};
            else
                shape_ = shape;
        }

        friend class multivariate_normal_distribution;

        std::vector<boost::random::normal_distribution<RealType>> distr_;
        std::vector<std::size_t> shape_;
    };

    // construct/copy/destruct
    multivariate_normal_distribution() : param_{} {}

    multivariate_normal_distribution(const param_type & param) : param_{param} {}

    template<class Iter>
    multivariate_normal_distribution(Iter mean_first, Iter mean_last, RealType covariance)
            : param_(mean_first, mean_last, covariance) {}

    template<class IterMean, class IterSigma>
    multivariate_normal_distribution(IterMean mean_first, IterMean mean_last, IterSigma covariance_first, IterSigma covariance_last)
            : param_(mean_first, mean_last, covariance_first, covariance_last) {}

    multivariate_normal_distribution(const std::initializer_list<RealType> & mean, RealType covariance)
            : param_(mean, covariance) {}


    multivariate_normal_distribution(const std::initializer_list<RealType> & mean,  const std::initializer_list<RealType> & covariance)
            : param_(mean, covariance) {}

    template<typename RangeMean>
    explicit multivariate_normal_distribution(const RangeMean & mean,  RealType covariance) : param_(mean, covariance) {}

    template<typename RangeMean, class RangeSigma>
    explicit multivariate_normal_distribution(const RangeMean & mean,  const RangeSigma & covariance) : param_(mean, covariance) {}

    multivariate_normal_distribution(const NDArray<RealType> & mean,  RealType covariance) : param_(mean, covariance) {}

    template<class IterSigma>
    multivariate_normal_distribution(const NDArray<RealType> & mean, IterSigma covariance_first, IterSigma covariance_last) : param_(mean, covariance_first, covariance_last) {}

    // public member functions
    NDArray<RealType> mean() const
    {
        return param_.mean();
    }

    std::vector<RealType> covariance() const
    {
        return param_.covariance();
    }

    std::vector<boost::random::normal_distribution<RealType>> distr() const
    {
        return param_.distr_;
    }

    std::vector<std::size_t> shape() const
    {
        return param_.shape();
    }

    param_type param() const
    {
        return param_;
    }

    void param(const param_type & param)
    {
        param_ = param;
    }

    void reset() {}

    template<typename URNG> result_type operator()(URNG & rng)
    {
        std::vector<RealType> ret;
        std::transform(param_.distr_.begin(), param_.distr_.end(), std::back_inserter(ret),
                       [&rng](auto & distr) { return distr(rng); });
        return NDArray<RealType>(std::move(ret), param_.shape_);
    }

    template<typename URNG> result_type operator()(URNG & rng, const param_type & param)
    {
        return multivariate_normal_distribution(param)(rng);
    }

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os,
               const multivariate_normal_distribution & distr)
    {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is,
               multivariate_normal_distribution & distr)
    {
        return is >> distr.param_;
    }

    friend bool operator==(const multivariate_normal_distribution & lhs,
                           const multivariate_normal_distribution & rhs)
    {
        return lhs.param_ == rhs.param_;
    }
    friend bool operator!=(const multivariate_normal_distribution & lhs,
                           const multivariate_normal_distribution & rhs)
    {
        return !(lhs == rhs);
    }

private:
    param_type param_;
};

} // end cpprob namespace
#endif //CPPROB_MULTIVARIATE_NORMAL_HPP_HPP
