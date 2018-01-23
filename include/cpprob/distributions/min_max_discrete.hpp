#ifndef INCLUDE_MIN_MAX_DISCRETE_HPP
#define INCLUDE_MIN_MAX_DISCRETE_HPP


#include <initializer_list>
#include <istream>
#include <iterator>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/random/discrete_distribution.hpp>

namespace cpprob {

template<typename IntType = int, typename WeightType = double>
class min_max_discrete_distribution {
public:
    using input_type  = WeightType;
    using result_type = IntType;

    class param_type {
    public:
        // types
        using distribution_type = min_max_discrete_distribution;

        // construct/copy/destruct
        template<typename Iter>
        param_type(IntType min, IntType max, Iter first, Iter last)
                : min_{min}, max_{max}, distr_(first, last) { }

        param_type(IntType min, IntType max, const std::initializer_list <WeightType> & w)
            : min_{min}, max_{max}, distr_(w) { }

        template<typename Range>
        explicit param_type(IntType min, IntType max, const Range & range)
                : min_{min}, max_{max}, distr_(range) { }

        template<typename Func>
        param_type(IntType min, IntType max, std::size_t nw, double xmin, double xmax, Func fw)
            : min_{min}, max_{max}, distr_(nw, xmin, xmax, fw) { }

        // public member functions
        IntType min() const
        {
            return min_;
        }

        IntType max() const
        {
            return max_;
        }

        std::vector <WeightType> probabilities() const
        {
            return distr_.probabilities();
        }

        boost::random::discrete_distribution<IntType, WeightType> distribution() const
        {
            return distr_;
        }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream <CharT, Traits> &
        operator<<(std::basic_ostream <CharT, Traits> & os, const param_type & param)
        {
            return os << param.min_ << " " << param.max_ << " " << param.dist_;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream <CharT, Traits> &
        operator>>(std::basic_istream <CharT, Traits> & is, param_type & param)
        {
            return is >> param.min_ >> std::ws >> param.max_ >> std::ws >> param.distr_;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return std::tie(lhs.min_, lhs.max_, lhs.distr_) ==
                   std::tie(rhs.min_, rhs.max_, rhs.distr_);
        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }
    private:
        friend min_max_discrete_distribution;
        IntType min_, max_;
        boost::random::discrete_distribution<IntType, WeightType> distr_;
    };

    // construct/copy/destruct
    template<typename Iter> min_max_discrete_distribution(IntType min, IntType max, Iter first, Iter last) : param_(min, max, first, last) {}

    min_max_discrete_distribution(IntType min, IntType max, const std::initializer_list<WeightType>& w) : param_(min, max, w) {}

    template<typename Range>
    explicit min_max_discrete_distribution(IntType min, IntType max, const Range & range) : param_(min, max, range) {}

    template<typename Func>
    min_max_discrete_distribution(
            IntType min,
            IntType max,
            std::size_t nw,
            double xmin,
            double xmax,
            Func fw) : param_(min, max, nw, xmin, xmax, fw) { }

    explicit min_max_discrete_distribution(const param_type & param) : param_(param) {}

    // public member functions
    template<typename URNG>
    IntType operator()(URNG & urng) const
    {
       return param_.min() + param_.distr_(urng);
    }

    template<typename URNG>
    IntType operator()(URNG & urng, const param_type & param) const
    {
        return param_.min() + param_.distr(urng, param.distr.param());
    }

    result_type min() const
    {
        return param_.min();
    }

    result_type max() const
    {
        return param_.max();
    }

    std::vector <WeightType> probabilities() const
    {
        return param_.distr_.probabilities();
    }

    param_type param() const
    {
        return param_;
    }

    void param(const param_type &param)
    {
        param_ = param;
    }

    void reset() {}

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream <CharT, Traits> &
    operator<<(std::basic_ostream <CharT, Traits> & os,
               const min_max_discrete_distribution & distr)
    {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream <CharT, Traits> &
    operator>>(std::basic_istream <CharT, Traits> & is,
               min_max_discrete_distribution & distr)
    {
        return is >> distr.param_;
    }


    friend bool operator==(const min_max_discrete_distribution & lhs,
                           const min_max_discrete_distribution & rhs )
    {
        return lhs.param_ == rhs.param_;
    }

    friend bool operator!=(const min_max_discrete_distribution & lhs,
                           const min_max_discrete_distribution & rhs)
    {
        return lhs.param_ != rhs.param_;
    }

private:
    param_type param_;
};


}  // end namespace cpprob
#endif //INCLUDE_MIN_MAX_DISCRETE_HPP
