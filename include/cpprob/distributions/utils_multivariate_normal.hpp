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
    using type = infcomp::protocol::MultivariateNormal;
};

template<class RealType>
struct serialise<multivariate_normal_distribution<RealType>> {
    using prior = multivariate_normal_distribution<RealType>;

    static proposal_t<prior> from_flatbuffers(const infcomp::protocol::ProposalReply *msg) {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());
        auto mean_ptr = distr->proposal_mean()->data()->begin();
        auto shape_ptr = distr->proposal_mean()->shape()->begin();
        auto sigma_ptr = distr->proposal_std()->data()->begin();
        auto dim = distr->proposal_mean()->data()->size();
        auto shape_size = distr->proposal_mean()->shape()->size();

        auto vec_data = std::vector<RealType>(mean_ptr, mean_ptr+dim);
        auto vec_shape = std::vector<int>(shape_ptr, shape_ptr+shape_size);
        return proposal_t<prior>(NDArray<RealType>(std::move(vec_data), std::move(vec_shape)), sigma_ptr, sigma_ptr + dim);
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff, const prior & distr) {
        auto mean = distr.mean();
        auto sigma = distr.sigma();
        auto mean_nd = NDArray<double>(mean.begin(), mean.end());
        auto sigma_nd = NDArray<double>(sigma.begin(), sigma.end());
        return infcomp::protocol::CreateMultivariateNormal(buff,
                                                           infcomp::protocol::CreateNDArray(buff,
                                                                                            buff.CreateVector<double>(mean_nd.values()),
                                                                                            buff.CreateVector<int32_t>(mean_nd.shape())),
                                                           infcomp::protocol::CreateNDArray(buff,
                                                                                            buff.CreateVector<double>(sigma_nd.values()),
                                                                                            buff.CreateVector<int32_t>(sigma_nd.shape()))
        ).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_MULTIVARIATE_NORMAL_HPP
