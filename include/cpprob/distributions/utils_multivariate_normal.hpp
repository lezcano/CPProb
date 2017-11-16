#ifndef CPPROB_UTILS_MULTIVARIATE_NORMAL_HPP
#define CPPROB_UTILS_MULTIVARIATE_NORMAL_HPP

#include <cmath>
#include <type_traits>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/distributions/utils_normal_distribution.hpp" // logpdf(normal)
#include "cpprob/ndarray.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<typename RealType>
struct logpdf<multivariate_normal_distribution<RealType>> {
    RealType operator()(const multivariate_normal_distribution<RealType>& distr,
                        const typename multivariate_normal_distribution<RealType>::result_type & x) const
    {
        RealType ret = 0;
        auto vec_distr = distr.distr();
        auto it_distr = vec_distr.begin();
        auto it_x = x.begin();
        for(; it_distr != vec_distr.end() && it_x != x.end(); ++it_distr, ++it_x)
            ret += logpdf<std::decay_t<decltype(*it_distr)>>()(*it_distr, *it_x);
        return ret;
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class RealType>
struct proposal<multivariate_normal_distribution<RealType>> {
    using type = multivariate_normal_distribution<RealType>;
};

template<class RealType>
struct buffer<multivariate_normal_distribution<RealType>> {
    using type = protocol::MultivariateNormal;
};

template<class RealType>
struct serialise<multivariate_normal_distribution<RealType>> {
    using prior = multivariate_normal_distribution<RealType>;

    static proposal_t<prior> from_flatbuffers(const protocol::ReplyProposal *msg) {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        return proposal_t<prior>(distr->mean()->begin(), distr->mean()->end(),
                                 distr->covariance()->begin(), distr->covariance()->end());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff,
                                                    const prior & distr,
                                                    const typename prior::result_type & value) {
        return protocol::CreateMultivariateNormal(buff,
                    buff.CreateVector<double>(distr.mean().values()),
                    buff.CreateVector<double>(distr.covariance()),
                    buff.CreateVector<double>(value.values())
        ).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_MULTIVARIATE_NORMAL_HPP
