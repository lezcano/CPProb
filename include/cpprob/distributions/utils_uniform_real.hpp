#ifndef CPPROB_UTILS_UNIFORM_REAL_HPP
#define CPPROB_UTILS_UNIFORM_REAL_HPP

#include <cmath>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/mixture.hpp"
#include "cpprob/distributions/truncated.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<typename RealType>
struct logpdf<boost::random::uniform_real_distribution<RealType>> {
    RealType operator()(const boost::random::uniform_real_distribution<RealType>& distr,
                        const typename boost::random::uniform_real_distribution<RealType>::result_type &) const
    {
        return -std::log(distr.b()-distr.a());
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class RealType>
struct proposal<boost::random::uniform_real_distribution<RealType>> {
    using type = mixture<truncated<boost::random::normal_distribution<RealType>>>;
};

template<class RealType>
struct buffer<boost::random::uniform_real_distribution<RealType>> {
    using type = protocol::MixtureTruncated;
};

template<class RealType>
struct serialise<boost::random::uniform_real_distribution<RealType>> {
    using prior = boost::random::uniform_real_distribution<RealType>;

    static proposal_t<prior> from_flatbuffers(const protocol::ReplyProposal *msg) {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());

        std::vector<truncated<boost::random::normal_distribution<RealType>>> v_distr(distr->distributions()->size());
        auto truncated_ptr = distr->distributions()->begin();
        for(std::size_t i = 0; i < distr->distributions()->size(); ++i) {
            auto normal_i = static_cast<const protocol::Normal *>(truncated_ptr->distribution());

            v_distr[i] = truncated<boost::normal_distribution<RealType>>(
                    boost::normal_distribution<RealType>(normal_i->mean(), normal_i->std()),
                    truncated_ptr->min(), truncated_ptr->max());

            ++truncated_ptr;
        }
        return proposal_t<prior>(distr->coefficients()->begin(), distr->coefficients()->end(),
                                 v_distr.begin(), v_distr.end());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff,
                                                    const prior & distr,
                                                    const typename prior::result_type value) {
        return protocol::CreateUniformContinuous(buff, distr.a(), distr.b(), value).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_REAL_HPP
