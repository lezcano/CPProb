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
////////// Proposal //////////
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

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class RealType>
struct proposal<boost::random::normal_distribution<RealType>> {
    using type = boost::random::normal_distribution<RealType>;
};

template<class RealType>
struct buffer<boost::random::normal_distribution<RealType>> {
    using type = protocol::Normal;
};

template<class RealType>
struct serialise<boost::random::normal_distribution<RealType>> {
    using prior = boost::random::normal_distribution<RealType>;

    static proposal_t<prior> from_flatbuffers(const protocol::ReplyProposal *msg)
    {
        auto distr = static_cast<const buffer_t<prior> *>(msg->distribution());
        return proposal_t<prior>(distr->mean(), distr->std());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff,
                                                    const prior & distr,
                                                    typename prior::result_type value)
    {
        return protocol::CreateNormal(buff, distr.mean(), distr.sigma(), value).Union();
    }
};


//////////////////////////////
///////// Truncated  /////////
//////////////////////////////
template<class RealType>
struct truncated_normaliser<boost::random::normal_distribution < RealType>> {
static RealType normalise(const boost::random::normal_distribution <RealType> &distr,
                          const RealType &min, const RealType &max) {
    using namespace boost::math;
    normal_distribution <RealType> normal(distr.mean(), distr.sigma());
    return cdf(normal, max) - cdf(normal, min);
}
};

}
#endif //CPPROB_NORMAL_DISTRIBUTION_UTILS_HPP
