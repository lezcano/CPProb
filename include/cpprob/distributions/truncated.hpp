#ifndef INCLUDE_TRUNCATED_HPP
#define INCLUDE_TRUNCATED_HPP

#include <algorithm>
#include <sstream> // ostringstream
#include <vector>

#include <boost/random/discrete_distribution.hpp>

// Truncate to [min, max]
template<class Distribution>
class truncated {
public:
    using input_type  = typename Distribution::input_type;
    using result_type = typename Distribution::result_type;

    class param_type {
    public:
        // types
        using distribution_type = truncated;

        // construct/copy/destruct
        param_type() = default;

        param_type(const Distribution & distr, const result_type & min, const result_type & max)
            : distr_(distr), min_(min), max_(max) { }

        // public member functions
        result_type min() const
        {
            return std::min(min_, distr_.min());
        }

        result_type max() const
        {
            return std::max(max_, distr_.max());
        }

        Distribution distribution() const
        {
            return distr_;
        }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream< CharT, Traits > &
        operator<<(std::basic_ostream< CharT, Traits > & os, const param_type & param)
        {
            return os << param.distr_ << os.widen(' ') << param.min_ << os.widen(' ') << param.max_;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream< CharT, Traits > &
        operator>>(std::basic_istream< CharT, Traits > & is, param_type & param)
        {
            return is >> param.distr_>> std::ws >> param.min_ >> std::ws >> param.max_;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return std::tie(lhs.distr_, lhs.min_, lhs.max_) ==
                   std::tie(rhs.distr_, rhs.min_, rhs.max_);
        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }

    private:
        friend truncated;

        Distribution distr_;
        result_type min_;
        result_type max_;
    };

    // construct/copy/destruct
    truncated() = default;

    truncated(const Distribution & distr, const result_type & min, const result_type & max)
            : param_(distr, min, max) {}

    explicit truncated(const param_type & param) : param_(param) {}

    // public member functions
    template<typename URNG>
    result_type operator()(URNG & urng)
    {
        result_type ret;
        for (std::size_t i = 0; i < 1'000'000; ++i) {
            ret = param_.distr_(urng);
            if (param_.min_ <= ret && ret <= param_.max_) {
                return ret;
            }
        }
        std::ostringstream os;
        os << "Error trying to sample an element from a distribution "
           << param_.distr_
           << " truncated in min: "
           << param_.min_
           << " and max: "
           << param_.max_
            << std::endl;
        throw std::runtime_error(os.str());
    }

    template<typename URNG>
    result_type operator()(URNG & urng, const param_type & param)
    {
        return truncated(param)(urng);
    }

    result_type min() const
    {
        return param_.min();
    }

    result_type max() const
    {
        return param_.max();
    }

    Distribution distribution() const
    {
        return param_.distribution();
    }

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const truncated & distr)
    {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is, truncated & distr)
    {
        return is >> distr.param_;
    }

    friend bool operator==(const truncated & lhs, const truncated & rhs)
    {
        return lhs.param_ == rhs.param_;
    }
    friend bool operator!=(const truncated & lhs, const truncated & rhs)
    {
        return !(lhs == rhs);
    }

private:
    param_type param_;
};

#endif //INCLUDE_TRUNCATED_HPP
