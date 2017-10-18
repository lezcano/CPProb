#ifndef CPPROB_UTILS_TRUNCATED_HPP
#define CPPROB_UTILS_TRUNCATED_HPP

#include <cmath>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/truncated.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template <class Distribution>
struct logpdf<truncated<Distribution>> {
    auto operator()(const truncated<Distribution>& distr,
                    const typename truncated<Distribution>::result_type & x) -> decltype(logpdf<Distribution>()(distr.distribution(), x)) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<decltype(logpdf<Distribution>()(distr.distribution(), x))>::infinity();
        }
        const auto underlying_distr = distr.distribution();
        return logpdf<Distribution>()(underlying_distr, x) -
               std::log(truncated_normaliser<Distribution>::normalise(underlying_distr, distr.min(), distr.max()));
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_TRUNCATED_HPP
