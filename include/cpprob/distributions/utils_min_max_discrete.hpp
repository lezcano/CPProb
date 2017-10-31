#ifndef CPPROB_UTILS_MIN_MAX_DISCRETE_HPP
#define CPPROB_UTILS_MIN_MAX_DISCRETE_HPP

#include <cmath>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/min_max_discrete.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<class IntType, class WeightType>
struct logpdf<min_max_discrete_distribution<IntType, WeightType>> {
    WeightType operator()(const min_max_discrete_distribution<IntType, WeightType>& distr,
                          const typename min_max_discrete_distribution<IntType, WeightType>::result_type & x) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<WeightType>::infinity();
        }
        return std::log(distr.probabilities()[static_cast<std::size_t>(x-distr.min())]);
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_MIN_MAX_DISCRETE_HPP
