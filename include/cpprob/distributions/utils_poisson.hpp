#ifndef CPPROB_UTILS_POISSON_HPP
#define CPPROB_UTILS_POISSON_HPP

#include <cmath>

#include <boost/random/poisson_distribution.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////// Prior & Proposal //////
//////////////////////////////

template<typename IntType, class RealType>
struct logpdf<boost::random::poisson_distribution<IntType, RealType>> {
    RealType operator()(const boost::random::poisson_distribution<IntType, RealType>& distr,
                        const typename boost::random::poisson_distribution<IntType, RealType>::result_type & x) const
    {

        if(std::numeric_limits<IntType>::has_infinity && std::abs(x) == std::numeric_limits<IntType>::infinity()) {
            return -std::numeric_limits<RealType>::infinity();
        }

        auto l = distr.mean();
        if (l == 0.0) {
            return -std::numeric_limits<RealType>::infinity();
        }
        RealType ret = x * std::log(l) - l;
        for (int i = 1; i <= x; ++i)
            ret -= std::log(i);
        return ret;
    }
};

template<class IntType, class RealType>
struct buffer<boost::random::poisson_distribution<IntType, RealType>> {
    using type = protocol::Poisson;
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class IntType, class RealType>
struct proposal<boost::random::poisson_distribution<IntType, RealType>> {
    using type = boost::random::poisson_distribution<IntType, RealType>;
};

template<class IntType, class RealType>
struct to_flatbuffers<boost::random::poisson_distribution<IntType, RealType>> {
    using distr_t = boost::random::poisson_distribution<IntType, RealType>;

    flatbuffers::Offset<void> operator()(flatbuffers::FlatBufferBuilder & buff,
                                         const distr_t & distr,
                                         const typename distr_t::result_type value)
    {
        return protocol::CreatePoisson(buff, distr.mean(), value).Union();
    }
};

//////////////////////////////
///////// Proposal  //////////
//////////////////////////////

template<class IntType, class RealType>
struct from_flatbuffers<boost::random::poisson_distribution<IntType, RealType>> {
    using distr_t = boost::random::poisson_distribution<IntType, RealType>;

    distr_t operator()(const buffer_t<distr_t> * distr) {
        return distr_t(distr->mean());
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_POISSON_HPP
