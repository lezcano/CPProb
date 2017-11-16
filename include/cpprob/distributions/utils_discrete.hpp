#ifndef CPPROB_UTILS_DISCRETE_HPP
#define CPPROB_UTILS_DISCRETE_HPP

#include <cmath>

#include <boost/random/discrete_distribution.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<class IntType, class WeightType>
struct logpdf <boost::random::discrete_distribution<IntType, WeightType>>{
    WeightType operator()(const boost::random::discrete_distribution<IntType, WeightType> &distr,
                          const typename boost::random::discrete_distribution<IntType, WeightType>::result_type &x) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<WeightType>::infinity();
        }
        return std::log(distr.probabilities()[static_cast<std::size_t>(x)]);
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class IntType, class WeightType>
struct proposal<boost::random::discrete_distribution<IntType, WeightType>> {
    using type = boost::random::discrete_distribution<IntType, WeightType>;
};

template<class IntType, class WeightType>
struct buffer<boost::random::discrete_distribution<IntType, WeightType>> {
    using type = protocol::Discrete;
};

template<class IntType, class WeightType>
struct serialise<boost::random::discrete_distribution<IntType, WeightType>> {
    using prior = boost::random::discrete_distribution<IntType, WeightType>;

    static proposal_t<prior> from_flatbuffers(const protocol::ReplyProposal *msg)
    {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        flatbuffers::Vector<double>::iterator probs_ptr = distr->probabilities()->begin();
        return proposal_t<prior>(probs_ptr, distr->probabilities()->end());

    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff,
                                                    const prior & distr,
                                                    typename prior::result_type value)
    {
        return protocol::CreateDiscrete(buff, 0, buff.CreateVector<double>(distr.probabilities()), value).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_DISCRETE_HPP
