#ifndef CPPROB_UTILS_POISSON_HPP
#define CPPROB_UTILS_POISSON_HPP

#include <cmath>

#include <boost/random/poisson_distribution.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
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
        if (l == 0) {
            return -std::numeric_limits<RealType>::infinity();
        }
        RealType ret = x * std::log(l) - l;
        for (int i = 1; i <= x; ++i)
            ret -= std::log(i);
        return ret;
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class IntType, class RealType>
struct proposal<boost::random::poisson_distribution<IntType, RealType>> {
    using type = boost::random::poisson_distribution<IntType, RealType>;
};

template<class IntType, class RealType>
struct buffer<boost::random::poisson_distribution<IntType, RealType>> {
    using type = infcomp::protocol::Poisson;
};

template<class IntType, class RealType>
struct serialise<boost::random::poisson_distribution<IntType, RealType>> {
    using prior = boost::random::poisson_distribution<IntType, RealType>;

    static proposal_t<prior> from_flatbuffers(const infcomp::protocol::ProposalReply *msg) {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        return proposal_t<prior>(distr->proposal_lambda());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff, const prior & distr) {
        return infcomp::protocol::CreatePoisson(buff, distr.mean()).Union();
    }
};


} // end namespace cpprob

#endif //CPPROB_UTILS_POISSON_HPP
