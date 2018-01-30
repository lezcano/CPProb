#ifndef CPPROB_UTILS_NORMAL_DISTRIBUTION_HPP
#define CPPROB_UTILS_NORMAL_DISTRIBUTION_HPP

#include <cmath>
#include <limits>

#include <boost/random/normal_distribution.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/normal.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////// Prior & Proposal //////
//////////////////////////////

template <class RealType>
struct logpdf<boost::random::normal_distribution<RealType>> {
    RealType operator()(const boost::random::normal_distribution<RealType>& distr,
                        const typename boost::random::normal_distribution<RealType>::result_type & x) const
    {
        RealType mean = distr.mean();
        RealType std = distr.sigma();

        if(std::numeric_limits<RealType>::has_infinity && std::abs(x) == std::numeric_limits<RealType>::infinity()) {
            return -std::numeric_limits<RealType>::infinity();
        }

        RealType result = (x - mean)/std;
        result *= result;
        result += std::log(2 * boost::math::constants::pi<RealType>() * std * std);
        result *= -0.5;

        return result;
    }
};

template<class RealType>
struct buffer<boost::random::normal_distribution<RealType>> {
    using type = protocol::Normal;
};


//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class RealType>
struct proposal<boost::random::normal_distribution<RealType>> {
    using type = boost::random::normal_distribution<RealType>;
};

template<class RealType>
struct to_flatbuffers<boost::random::normal_distribution<RealType>> {
    using distr_t = boost::random::normal_distribution<RealType>;

    flatbuffers::Offset<void> operator()(flatbuffers::FlatBufferBuilder & buff,
                                         const distr_t & distr,
                                         typename distr_t::result_type value)
    {
        return protocol::CreateNormal(buff, distr.mean(), distr.sigma(), value).Union();
    }
};

//////////////////////////////
///////// Proposal  //////////
//////////////////////////////

template<class RealType>
struct from_flatbuffers<boost::random::normal_distribution<RealType>> {
    using distr_t = boost::random::normal_distribution<RealType>;

    distr_t operator()(const buffer_t<distr_t> * distr)
    {
        return distr_t(distr->mean(), distr->std());
    }
};


//////////////////////////////
///////// Truncated  /////////
//////////////////////////////

template<class RealType>
struct normalise<boost::random::normal_distribution<RealType>> {
    RealType operator()(const boost::random::normal_distribution<RealType> & distr,
                        const RealType & min, const RealType & max)
    {
        boost::math::normal_distribution <RealType> normal(distr.mean(), distr.sigma());
        return boost::math::cdf(normal, max) - boost::math::cdf(normal, min);
    }
};

}
#endif //CPPROB_NORMAL_DISTRIBUTION_UTILS_HPP
