#ifndef CPPROB_MULTIVARIATE_NORMAL_HPP_HPP
#define CPPROB_MULTIVARIATE_NORMAL_HPP_HPP

#include <vector>
#include <iterator>
#include <initializer_list>
#include <algorithm>
#include <tuple> // std::tie

#include <boost/random/normal_distribution.hpp>

namespace cpprob {

template<typename RealType = double>
class multivariate_normal_distribution {
public:
    // types
    typedef std::vector<RealType> input_type;
    typedef std::vector<RealType> result_type;

    // member classes/structs/unions

    class param_type {
    public:

        // types
        using distribution_type = multivariate_normal_distribution;

        // construct/copy/destruct
        param_type() : distr_{boost::random::normal_distribution<RealType>(0, 1)} {};

        template<class Iter>
        param_type(Iter mean_first, Iter mean_last, RealType sigma)
        {
            init(mean_first, mean_last, sigma);
        }

        template<class IterMean, class IterSigma>
        param_type(IterMean mean_first, IterMean mean_last, IterSigma sigma_first, IterSigma sigma_last)
        {
            init(mean_first, mean_last, sigma_first, sigma_last);
        }

        param_type(const std::initializer_list<RealType> & mean, RealType sigma)
        {
            init(mean.begin(), mean.end(),sigma);
        }

        param_type(const std::initializer_list<RealType> & mean,  const std::initializer_list<RealType> & sigma)
        {
            init(mean.begin(), mean.end(), sigma.begin(), sigma.end());
        }

        template<typename RangeMean>
        explicit param_type(const RangeMean & mean,  RealType sigma)
        {
            init(mean.begin(), mean.end(), sigma);
        }

        template<typename RangeMean, class RangeSigma>
        explicit param_type(const RangeMean & mean,  const RangeSigma & sigma)
        {
            init(mean.begin(), mean.end(), sigma.begin(), sigma.end());
        }

        // public member functions
        std::vector<RealType> mean() const
        {
            std::vector<RealType> mean;
            std::transform(distr_.begin(), distr_.end(), std::back_inserter(mean), [](const auto & distr) { return distr.mean(); });
            return mean;
        }

        std::vector<RealType> sigma() const
        {
            std::vector<RealType> sigma;
            std::transform(distr_.begin(), distr_.end(), std::back_inserter(sigma), [](const auto & distr) { return distr.sigma(); });
            return sigma;
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
            detail::print_vector(os, param.mean());
            os << os.widen(' ');
            detail::print_vector(os, param.sigma());
            return os;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream< CharT, Traits > &
        operator>>(std::basic_istream< CharT, Traits > & is, param_type &  param)
        {
            std::vector<RealType> mean_temp;
            detail::read_vector(is, mean_temp);
            if(!is) {
                return is;
            }

            is >> std::ws;

            std::vector<RealType> sigma_temp;
            detail::read_vector(is, sigma_temp);
            if(!is) {
                return is;
            }

            if (sigma_temp == 1){
                param.init(mean_temp.begin(), mean_temp.end(), sigma_temp.front());
                return is;
            }

            param.init(mean_temp.begin(), mean_temp.end(), sigma_temp.begin(), sigma_temp.end());

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
        void init(Iter mean_first, Iter mean_last, RealType sigma)
        {
            distr_.clear();
            for (; mean_first != mean_last; ++mean_first)
                distr_.emplace_back(*mean_first, sigma);
        }

        template<class IterMean, class IterSigma>
        void init(IterMean mean_first, IterMean mean_last, IterSigma sigma_first, IterSigma sigma_last)
        {
            for (; mean_first != mean_last && sigma_first != sigma_last; ++mean_first, ++sigma_first)
                distr_.emplace_back(*mean_first, *sigma_first);
        }

        friend class multivariate_normal_distribution;

        std::vector<boost::random::normal_distribution<RealType>> distr_;
    };

    // construct/copy/destruct
    multivariate_normal_distribution() : param_{} {}

    template<class Iter>
    multivariate_normal_distribution(Iter mean_first, Iter mean_last, RealType sigma)
            : param_(mean_first, mean_last, sigma) {}

    template<class IterMean, class IterSigma>
    multivariate_normal_distribution(IterMean mean_first, IterMean mean_last, IterSigma sigma_first, IterSigma sigma_last)
            : param_(mean_first, mean_last, sigma_first, sigma_last) {}

    multivariate_normal_distribution(const std::initializer_list<RealType> & mean, RealType sigma)
            : param_(mean, sigma) {}


    multivariate_normal_distribution(const std::initializer_list<RealType> & mean,  const std::initializer_list<RealType> & sigma)
            : param_(mean, sigma) {}

    template<typename RangeMean>
    explicit multivariate_normal_distribution(const RangeMean & mean,  RealType sigma) : param_(mean, sigma) {}

    template<typename RangeMean, class RangeSigma>
    explicit multivariate_normal_distribution(const RangeMean & mean,  const RangeSigma & sigma) : param_(mean, sigma) { }

    // public member functions
    std::vector<RealType> mean() const
    {
        return param_.mean();
    }

    std::vector<RealType> sigma() const
    {
        return param_.sigma();
    }

    std::vector<boost::random::normal_distribution<RealType>> distr() const
    {
        return param_.distr_;
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
        return ret;
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
