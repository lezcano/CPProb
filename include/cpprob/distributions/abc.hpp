#ifndef CPPROB_ABC_HPP
#define CPPROB_ABC_HPP

#include <type_traits>

#include "cpprob/utils.hpp"
#include "cpprob/traits.hpp"

namespace cpprob {

template<class Sample, class LogPDF>
class ABC {
public:
    using input_type = void;
    // TODO(Lezcano) This should be std::invoke_result_t in C++17
    using result_type = std::result_of_t<Sample>;

    class param_type {
    public:
        // types
        using distribution_type = ABC;

        // construct/copy/destruct
        param_type() = default;

        param_type(const Sample & sample, const LogPDF & logpdf) : sample_(sample), logpdf_(logpdf) {}

        result_type operator()()
        {
            return sample_();
        }

        template<typename URNG>
        result_type operator()(URNG &)
        {
            return sample_();
        }

        double logpdf(const result_type & x) const
        {
            return logpdf_(x);
        }

        // friend functions
        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return std::tie(lhs.sample_, lhs.logpdf_) == std::tie(rhs.sample_, rhs.logpdf_);
        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }

    private:
        friend ABC;

        Sample sample_;
        LogPDF logpdf_;
    };

    ABC() = default;

    ABC(const Sample & sample, const LogPDF & logpdf) : param_(sample, logpdf) {}

    explicit ABC(const param_type & param) : param_(param) {}

    result_type operator()()
    {
        return param_();
    }

    template<typename URNG>
    result_type operator()(URNG &)
    {
        return param_();
    }

    template<typename URNG>
    result_type operator()(URNG &, const param_type & param)
    {
        return ABC(param)();
    }

    double logpdf(const result_type & x) const
    {
        return param_.logpdf_(x);
    }

    // friend functions
    friend bool operator==(const param_type & lhs, const param_type & rhs)
    {
        return lhs.param_ == rhs.param_;
    }

    friend bool operator!=(const param_type & lhs, const param_type & rhs)
    {
        return !(lhs == rhs);
    }

private:
    param_type param_;
};

} // end namespace cpprob

#endif //CPPROB_ABC_HPP
