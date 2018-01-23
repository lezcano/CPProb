#ifndef INCLUDE_DIRAC_DELTA_HPP
#define INCLUDE_DIRAC_DELTA_HPP

#include <istream>
#include <ostream>
#include <type_traits>
#include <utility>

namespace cpprob {

template<typename T>
class dirac_delta {
public:
    using input_type  = std::decay_t<T>;
    using result_type = std::decay_t<T>;

    class param_type {
    public:
        // types
        using distribution_type = dirac_delta;

        // construct/copy/destruct
        param_type(const input_type & x) : x_(x) {}
        param_type(input_type && x) : x_(std::move(x)) {}

        // public member functions
        template<typename URNG>
        result_type operator()(URNG &) const { return x_; }
        result_type operator()() const { return x_; }

        result_type x() const { return x_; }
        result_type min() const { return x_; }
        result_type max() const { return x_; }

        // friend functions
        template<typename CharT, typename Traits>
        friend std::basic_ostream <CharT, Traits> &
        operator<<(std::basic_ostream <CharT, Traits> & os, const param_type & param)
        {
            return os << param.x_;
        }

        template<typename CharT, typename Traits>
        friend std::basic_istream <CharT, Traits> &
        operator>>(std::basic_istream <CharT, Traits> & is, param_type & param)
        {
            return is >> param.x_;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return lhs.x_ == rhs.x_;
        }

        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }
    private:
        friend dirac_delta;
        result_type x_;
    };

    // construct/copy/destruct
    dirac_delta(const input_type & x) : param_(x) {}
    dirac_delta(input_type && x) : param_(std::move(x)) {}

    explicit dirac_delta(const param_type & param) : param_(param) {}
    explicit dirac_delta(param_type && param) : param_(std::move(param)) {}

    // public member functions
    template<typename URNG>
    result_type operator()(URNG &) const { return param_(); }
    result_type operator()() const { return param_(); }

    template<typename URNG>
    result_type operator()(URNG & urng, const param_type & param) const
    {
        return param.x_;
    }

    result_type x() const { return param_.x_; }

    result_type min() const { return param_.x_; }

    result_type max() const { return param_.x_; }

    param_type param() const
    {
        return param_;
    }

    void param(const param_type & param) { param_ = param; }
    void param(param_type && param) { param_ = std::move(param); }

    void reset() {}

    // friend functions
    template<typename CharT, typename Traits>
    friend std::basic_ostream <CharT, Traits> &
    operator<<(std::basic_ostream <CharT, Traits> & os, const dirac_delta & distr)
    {
        return os << distr.param_;
    }

    template<typename CharT, typename Traits>
    friend std::basic_istream <CharT, Traits> &
    operator>>(std::basic_istream <CharT, Traits> & is, dirac_delta & distr)
    {
        return is >> distr.param_;
    }


    friend bool operator==(const dirac_delta & lhs, const dirac_delta & rhs )
    {
        return lhs.param_ == rhs.param_;
    }

    friend bool operator!=(const dirac_delta & lhs, const dirac_delta & rhs)
    {
        return lhs.param_ != rhs.param_;
    }

private:
    param_type param_;
};

template<class T>
auto make_dirac_delta(T && x)
{
    return dirac_delta<T>(x);
}

}  // end namespace cpprob
#endif //DIRAC_DELTA_HPP
