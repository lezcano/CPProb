#ifndef CPPROB_ABC_HPP
#define CPPROB_ABC_HPP

#include <tuple>
#include <type_traits>

#include "cpprob/distributions/utils_base.hpp" // forward declare cpprob::logpdf
#include "cpprob/traits.hpp"
#include "cpprob/utils.hpp"

namespace cpprob {
namespace detail {
template<class Sample, class URNG>
auto sample_impl(Sample & sample, URNG & urng, std::true_type)
{
    return sample(urng);
}

template<class Sample, class URNG>
auto sample_impl(Sample & sample, URNG &, std::false_type)
{
    return sample();
}

template<class LogPDF, class result_type>
auto logpdf_impl(const LogPDF & logpdf, const result_type & x, std::true_type)
{
    return cpprob::logpdf<LogPDF>()(logpdf, x);
}

template<class LogPDF, class result_type>
auto logpdf_impl(const LogPDF & logpdf, const result_type & x, std::false_type)
{
    return logpdf(x);
}

template<class Sample>
typename Sample::input_type input_type_impl(std::true_type);
template<class Sample>
void input_type_impl(std::false_type);

template<class Sample>
typename Sample::result_type result_type_impl(std::true_type);
// TODO(Lezcano) This should be std::invoke_result_t in C++17
template<class Sample>
std::result_of_t<Sample> result_type_impl(std::false_type);
} // end namespace detail

template<class Sample, class LogPDF>
class ABC {
public:
    using sample_t = std::decay_t<Sample>;
    using logpdf_t = std::decay_t<LogPDF>;

    using input_type = decltype(detail::input_type_impl<sample_t>(cpprob::is_distribution<sample_t>{}));
    using result_type = decltype(detail::result_type_impl<sample_t>(cpprob::is_distribution<sample_t>{}));

    class param_type {
    public:
        // types
        using distribution_type = ABC;

        // construct/copy/destruct
        param_type() = default;

        param_type(Sample && sample, LogPDF && logpdf) : sample_(std::forward<Sample>(sample)),
                                                         logpdf_(std::forward<LogPDF>(logpdf)) {}

        template<class URNG>
        auto operator()(URNG & urng)
        {
            return detail::sample_impl<sample_t>(sample_, urng, cpprob::is_distribution<sample_t>{});
        }

        auto logpdf(const result_type & x) const
        {
            return detail::logpdf_impl<logpdf_t, result_type>(logpdf_, x, cpprob::is_distribution<logpdf_t>{});
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

        sample_t sample_;
        logpdf_t logpdf_;
    };

    ABC() = default;

    ABC(Sample && sample, LogPDF && logpdf) : param_(std::forward<Sample>(sample), std::forward<LogPDF>(logpdf)) {}

    explicit ABC(const param_type & param) : param_(param) {}

    template<class URNG>
    result_type operator()(URNG & urng)
    {
        return param_(urng);
    }

    template<class URNG>
    result_type operator()(URNG & urng, const param_type & param)
    {
        return ABC(param)(urng);
    }

    auto logpdf(const result_type & x) const
    {
        return param_.logpdf(x);
    }

    // friend functions
    friend bool operator==(const ABC & lhs, const ABC & rhs)
    {
        return lhs.param_ == rhs.param_;
    }

    friend bool operator!=(const ABC & lhs, const ABC & rhs)
    {
        return !(lhs == rhs);
    }

private:
    param_type param_;
};

template<class Sample, class LogPDF>
auto make_abc(Sample && sample, LogPDF && logpdf)
{
    return ABC<Sample, LogPDF>(std::forward<Sample>(sample), std::forward<LogPDF>(logpdf));
}

} // end namespace cpprob

#endif //CPPROB_ABC_HPP
