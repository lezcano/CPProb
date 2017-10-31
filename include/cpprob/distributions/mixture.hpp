#ifndef INCLUDE_MIXTURE_HPP
#define INCLUDE_MIXTURE_HPP

#include <algorithm>
#include <initializer_list>
#include <vector>

#include <boost/random/discrete_distribution.hpp>

#include "cpprob/serialization.hpp"

template<class Distribution, class RealType=double>
class mixture {
public:
    using input_type  = typename Distribution::input_type;
    using result_type = typename Distribution::result_type;

    class param_type {
    public:
        // types
        using distribution_type = mixture;

        // construct/copy/destruct
        param_type() = default;

        template<class IterCoef, class IterDistr>
        param_type(IterCoef coef_first, IterCoef coef_last, IterDistr distr_first, IterDistr distr_last)
            : coef_(coef_first, coef_last), distr_(distr_first, distr_last) { }

        param_type(const std::initializer_list <RealType> & coef, const std::initializer_list<Distribution> & distr)
            : coef_(coef), distr_(distr.begin(), distr.end()) { }

        template<class RangeCoef, class RangeDistr>
        explicit param_type(const RangeCoef & coef, const RangeDistr & distr)
                : coef_(coef), distr_(distr.begin(), distr.end()) { }

        // public member functions
        // Implying that Distribution has min and max -> Maybe SFINAE these out if not?
        result_type min() const
        {
            return std::min_element(distr_.begin(), distr_.end(),
                                    [](const Distribution & lhs, const Distribution & rhs){
                                        return lhs.min() < rhs.min();
                                    })->min();
        }

        result_type max() const
        {
            return std::min_element(distr_.begin(), distr_.end(),
                                    [](const Distribution & lhs, const Distribution & rhs){
                                        return lhs.max() < rhs.max();
                                    })->max();
        }

        std::vector<RealType> coefficients() const
        {
            return coef_.probabilities();
        }

        std::vector<Distribution> distributions() const
        {
            return distr_;
        }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream< CharT, Traits > &
        operator<<(std::basic_ostream< CharT, Traits > & os, const param_type & param)
        {
            return os << param.coef_ << os.widen(' ') << param.distr_;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream< CharT, Traits > &
        operator>>(std::basic_istream< CharT, Traits > & is, param_type & param)
        {
            std::vector<RealType> v;
            is >> param.coef_>> std::ws >> v;
            param.distr_ = boost::random::discrete_distribution<std::size_t, RealType>(v.begin(), v.end());
            return is;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return std::tie(lhs.coef_, lhs.distr_) == std::tie(rhs.coef_, rhs.distr_);
        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }

    private:
        friend mixture;

        boost::random::discrete_distribution<std::size_t, RealType> coef_;
        std::vector<Distribution> distr_;
    };

    // construct/copy/destruct
    mixture() = default;

    template<class IterCoef, class IterDistr>
    mixture(IterCoef coef_first, IterCoef coef_last, IterDistr distr_first, IterDistr distr_last)
        : param_(coef_first, coef_last, distr_first, distr_last) {}

    mixture(const std::initializer_list <RealType> & coef, const std::initializer_list<Distribution> & distr)
        : param_(coef, distr) {}

    template<class RangeCoef, class RangeDistr>
    explicit mixture(const RangeCoef & coef, const RangeDistr & distr)
        : param_(coef, distr) {}

    explicit mixture(const param_type & param) : param_(param) {}

    // public member functions
    template<typename URNG>
    result_type operator()(URNG & urng)
    {
        return param_.distr_[param_.coef_(urng)](urng);
    }

    template<typename URNG>
    result_type operator()(URNG & urng, const param_type & param)
    {
        return mixture(param)(urng);
    }

    result_type min() const
    {
        return param_.min();
    }

    result_type max() const
    {
        return param_.max();
    }

    std::vector<RealType> coefficients() const
    {
        return param_.coefficients();
    }

    std::vector<Distribution> distributions() const
    {
        return param_.distributions();
    }

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const mixture & distr)
    {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is, mixture & distr)
    {
        return is >> distr.param_;
    }

    friend bool operator==(const mixture & lhs, const mixture & rhs)
    {
        return lhs.param_ == rhs.param_;
    }
    friend bool operator!=(const mixture & lhs, const mixture & rhs)
    {
        return !(lhs == rhs);
    }

private:
    param_type param_;
};

#endif //INCLUDE_MIXTURE_HPP
