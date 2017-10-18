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
    using type = infcomp::protocol::UniformDiscrete;
};

template<class IntType>
struct serialise<boost::random::uniform_smallint<IntType>> {
    using prior = boost::random::uniform_smallint<IntType>;

    static proposal_t<prior> from_flatbuffers(const infcomp::protocol::ProposalReply *msg)
    {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        flatbuffers::Vector<double>::iterator probs_ptr = distr->proposal_probabilities()->data()->begin();
        return proposal_t<prior>(distr->prior_min(),
                                 distr->prior_min() + distr->prior_size() - 1,
                                 probs_ptr, probs_ptr+distr->prior_size());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff, const prior & distr)
    {
        return infcomp::protocol::CreateUniformDiscrete(buff,distr.a(), distr.b()-distr.a()+1).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_SMALLINT_HPP
