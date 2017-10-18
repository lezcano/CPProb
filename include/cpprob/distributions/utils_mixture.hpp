#ifndef CPPROB_UTILS_MIXTURE_HPP
#define CPPROB_UTILS_MIXTURE_HPP

#include <cmath>

#include "cpprob/distributions/mixture.hpp"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<typename Distribution, class RealType>
struct logpdf<mixture<Distribution, RealType>> {
    RealType operator()(const mixture<Distribution, RealType>& distr,
                        const typename mixture<Distribution, RealType>::result_type & x) const
    {
        const auto coefs = distr.coefficients();
        auto coefs_first = coefs.cbegin();
        auto coefs_last = coefs.cend();
        const auto vec_distr = distr.distributions();
        auto vec_distr_first = vec_distr.cbegin();

        RealType ret = 0;

        while (coefs_first != coefs_last) {
            ret += *coefs_first*logpdf<std::decay_t<decltype(*vec_distr_first)>>()(*vec_distr_first, x);
            ++vec_distr_first;
            ++coefs_first;
        }
        return ret;
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_MIXTURE_HPP
