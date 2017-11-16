#ifndef CPPROB_UTILS_UNIFORM_SMALLINT_HPP
#define CPPROB_UTILS_UNIFORM_SMALLINT_HPP

#include <cmath>

#include <boost/random/uniform_smallint.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////
template <class IntType>
struct logpdf<boost::random::uniform_smallint<IntType>> {
    double operator()(const boost::random::uniform_smallint<IntType>& distr,
                      const typename boost::random::uniform_smallint<IntType>::result_type &) const
    {
        return -std::log(distr.max() - distr.min() + 1.0);
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////
template<class IntType>
struct proposal<boost::random::uniform_smallint<IntType>> {
    using type = min_max_discrete_distribution<IntType, double>;
};

template<class IntType>
struct buffer<boost::random::uniform_smallint<IntType>> {
    using type = protocol::Discrete;
};

template<class IntType>
struct serialise<boost::random::uniform_smallint<IntType>> {
    using prior = boost::random::uniform_smallint<IntType>;

    static proposal_t<prior> from_flatbuffers(const protocol::ReplyProposal *msg)
    {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        return proposal_t<prior>(distr->min(),
                                 distr->min() + distr->probabilities()->size() - 1,
                                 distr->probabilities()->begin(), distr->probabilities()->end());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff,
                                                    const prior & distr,
                                                    const typename prior::result_type value)
    {
        return protocol::CreateUniformDiscrete(buff,distr.a(), distr.b(), value).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_SMALLINT_HPP
