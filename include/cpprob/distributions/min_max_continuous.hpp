#ifndef CPPROB_MIN_MAX_CONTINUOUS_HPP
#define CPPROB_MIN_MAX_CONTINUOUS_HPP

#include <istream>
#include <ostream>
#include <tuple>

#include <boost/random/beta_distribution.hpp>

namespace cpprob {

template<class RealType = double>
class min_max_continuous_distribution {
public:
    // types
    using result_type = RealType;
    using input_type = RealType;

    // member classes/structs/unions
    class param_type {
    public:
        // types
        using distribution_type = min_max_continuous_distribution;

        // construct/copy/destruct
        // K + 2 = pdf(mode)^2 ; K \in R^+
        // http://doingbayesiandataanalysis.blogspot.co.uk/2012/06/beta-distribution-parameterized-by-mode.html
        explicit param_type(RealType min = 0.0, RealType max = 1.0, RealType mode = 0.5, RealType k = 0.0)
                : min_{min}, max_{max} {
            RealType normalised_mode = (mode - min) / (max - min);

            distr_ = boost::random::beta_distribution<RealType>(normalised_mode * k + 1, (1 - normalised_mode) * k + 1);
        }

        // public member functions

        RealType min() const {
            return min_;
        }

        RealType max() const {
            return max_;
        }

        RealType mode() const {
            auto a = distr_.alpha();
            auto b = distr_.beta();
            return min_ + (max_ - min_) * ((a - 1) / (a + b - 2));
        }

        RealType k() const {
            return distr_.alpha() + distr_.beta() - 2;
        }

        boost::random::beta_distribution<RealType> beta() const {
            return distr_;
        }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream<CharT, Traits> &
        operator<<(std::basic_ostream<CharT, Traits> &os, const param_type &param) {
            return os << param.min_ << " " << param.max_ << " " << param.dist_;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream<CharT, Traits> &
        operator>>(std::basic_istream<CharT, Traits> &is, const param_type &param) {
            return is >> param.min_ >> std::ws >> param.max_ >> std::ws >> param.distr_;
        }

        friend bool operator==(const param_type &lhs, const param_type &rhs) {
            return std::tie(lhs.min_, lhs.max_, lhs.distr_) ==
                   std::tie(rhs.min_, rhs.max_, rhs.distr_);
        }

        friend bool operator!=(const param_type &lhs, const param_type &rhs) {
            return !(lhs == rhs);
        }

    private:
        // Friend classes
        friend class min_max_continuous_distribution;

        // parameters
        RealType min_;
        RealType max_;
        boost::random::beta_distribution<RealType> distr_;
    };

    // construct/copy/destruct
    explicit min_max_continuous_distribution(RealType min = 0.0, RealType max = 1.0, RealType mode = 1.0,
                                             RealType k = 1.0)
            : param_{min, max, mode, k} {}

    explicit min_max_continuous_distribution(const param_type &param)
            : param_{param} {}

    // public member functions
    template<typename URNG>
    RealType operator()(URNG &rng) const {
        return param_.min_ + (param_.max_ - param_.min_) * param_.distr_(rng);
    }

    template<typename URNG>
    RealType operator()(URNG &rng, const param_type &param) const {
        return min_max_continuous_distribution(param)(rng);
    }

    RealType min() const {
        return param_.min_;
    }

    RealType max() const {
        return param_.max_;
    }

    RealType mode() const {
        return param_.k();
    }

    RealType k() const {
        return param_.k();
    }

    boost::random::beta_distribution<RealType> beta() const {
        return param_.distr_;
    }

    param_type param() const {
        return param_;
    }

    void param(const param_type &param) {
        param_ = param;
    }

    void reset() {}

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits> &
    operator<<(std::basic_ostream<CharT, Traits> &os,
               const min_max_continuous_distribution &distr) {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream<CharT, Traits> &
    operator>>(std::basic_istream<CharT, Traits> &is,
               min_max_continuous_distribution &distr) {
        return is >> distr.param_;
    }

    friend bool operator==(const min_max_continuous_distribution &lhs, const min_max_continuous_distribution &rhs) {
        return lhs.param_ == rhs.param_;
    }

    friend bool operator!=(const min_max_continuous_distribution &lhs, const min_max_continuous_distribution &rhs) {
        return lhs.param_ != rhs.param_;
    }

private:
    param_type param_;
};

} // end namespace cpprob

#endif  // CPPROB_MIN_MAX_CONTINUOUS_HPP
