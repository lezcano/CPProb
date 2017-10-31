#ifndef INCLUDE_MIN_MAX_CONTINUOUS_HPP
#define INCLUDE_MIN_MAX_CONTINUOUS_HPP

#include <cmath>

#include <boost/math/distributions/beta.hpp>

#include "cpprob/distributions/utils_base.hpp"

#include "cpprob/distributions/min_max_continuous.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////
template<class RealType>
struct logpdf<min_max_continuous_distribution<RealType>> {
    RealType operator()(const min_max_continuous_distribution<RealType> & distr,
                        const typename min_max_continuous_distribution<RealType>::result_type & x) const
    {
        auto beta_distr = distr.beta();
        auto a = beta_distr.alpha();
        auto b = beta_distr.beta();
        auto min = distr.min();
        auto max = distr.max();
        auto norm_x = (x - min) / (max - min);
        if (norm_x < 0 || norm_x > 1) {
            return -std::numeric_limits<RealType>::infinity();
        }

        return std::log(boost::math::pdf(boost::math::beta_distribution<RealType>(a, b), norm_x))
               - std::log(max - min);
    }
};

} // end namespace cpprob

#endif //INCLUDE_MIN_MAX_CONTINUOUS_HPP
