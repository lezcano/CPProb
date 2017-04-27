#ifndef INCLUDE_TRAITS_HPP_
#define INCLUDE_TRAITS_HPP_

#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <boost/random/uniform_smallint.hpp>

#include <boost/math/constants/constants.hpp>

#include "distr/min_max_discrete.hpp"
#include "distr/vmf.hpp"

#include "flatbuffers/infcomp_generated.h"

namespace cpprob {
namespace detail {

template<typename IntType = int, typename WeightType = double>
min_max_discrete_distribution<IntType, WeightType>
get_min_max_discrete(const infcomp::ProposalReply* msg)
{
    auto param = static_cast<const infcomp::UniformDiscrete*>(msg->distribution());
    flatbuffers::Vector<double>::iterator probs_ptr = param->proposal_probabilities()->data()->begin();
    return min_max_discrete_distribution<IntType, WeightType>(param->prior_min(),
                                                              param->prior_min() + param->prior_size() - 1,
                                                              probs_ptr, probs_ptr+param->prior_size());
}

} // end namespace detail

template <class RealType>
RealType pdf(const boost::random::normal_distribution<RealType>& distr,
             const typename boost::random::normal_distribution<RealType>::result_type & x)
{
    using namespace boost::math;
    return pdf(normal_distribution<RealType>{distr.mean(), distr.sigma()}, x);
}

template <class IntType>
double pdf(const boost::random::uniform_smallint<IntType>& distr,
           const typename boost::random::uniform_smallint<IntType>::result_type & x)
{
    return 1.0/(distr.max() - distr.min() + 1.0);
}

template<typename IntType, typename WeightType>
WeightType pdf(const min_max_discrete_distribution<IntType, WeightType>& dist,
               const typename min_max_discrete_distribution<IntType, WeightType>::result_type & x)
{
    if (x < dist.min() || x > dist.max()) return 0;
    return dist.probabilities()[static_cast<std::size_t>(x-dist.min())];
}

template<class RealType>
RealType pdf(const vmf_distribution<RealType>& distr,
             const typename vmf_distribution<RealType>::result_type & x)
{
    if (distr.kappa() == 0){
        return 1.0/(4.0 * boost::math::constants::pi<RealType>());
    }
    return 0;

}


template <class RealType>
boost::normal_distribution<>::param_type default_type_param(const boost::random::normal_distribution<RealType>& distr)
{
    return boost::normal_distribution<>::param_type(static_cast<double>(distr.mean()),
                                                    static_cast<double>(distr.sigma()));
}

template <class IntType>
boost::uniform_smallint<>::param_type default_type_param(const boost::random::uniform_smallint<IntType>& distr)
{
    return boost::uniform_smallint<>::param_type(static_cast<int>(distr.a()),
                                                 static_cast<int>(distr.b()));
}

template<class RealType>
vmf_distribution<>::param_type default_type_param(const vmf_distribution<RealType>& distr)
{
    auto mu = distr.mu();
    std::vector<double> mu_double(mu.begin(), mu.end());
    return vmf_distribution<>::param_type(mu_double, distr.kappa());
}

template<template <class ...> class Distr, class ...Params>
struct proposal { };

template<class RealType>
struct proposal<boost::random::normal_distribution, RealType> {
    static constexpr auto type_enum = infcomp::Distribution::Normal;

    static flatbuffers::Offset<void> request(flatbuffers::FlatBufferBuilder& fbb,
            const boost::random::normal_distribution<RealType>& distr)
    {
        return infcomp::CreateNormal(fbb, distr.mean(), distr.sigma()).Union();
    }

    static boost::random::normal_distribution<RealType>
    get_distr(const infcomp::ProposalReply* msg)
    {
        auto param = static_cast<const infcomp::Normal*>(msg->distribution());
        return boost::random::normal_distribution<RealType>(param->proposal_mean(), param->proposal_std());
    }
};


template<class IntType>
struct proposal<boost::random::uniform_smallint, IntType> {
    static constexpr auto type_enum = infcomp::Distribution::UniformDiscrete;

    static flatbuffers::Offset<void> request(flatbuffers::FlatBufferBuilder& fbb,
                                             const boost::random::uniform_smallint<IntType>& distr)
    {
        auto size = distr.max() - distr.min() + 1;
        return infcomp::CreateDiscrete(fbb, distr.min(), size).Union();
    }

    template<class RealType = double>
    static min_max_discrete_distribution<IntType, RealType>
    get_distr(const infcomp::ProposalReply* msg)
    {
        return detail::get_min_max_discrete<IntType, RealType>(msg);
    }
};

template<class RealType>
struct proposal<vmf_distribution, RealType> {
    static constexpr auto type_enum = infcomp::Distribution::VMF;

    static flatbuffers::Offset<void> request(flatbuffers::FlatBufferBuilder& fbb,
                                             const vmf_distribution<RealType>&)
    {
        throw 1;
        //return infcomp::Create(fbb).Union();
    }

    static vmf_distribution<RealType>
    get_distr(const infcomp::ProposalReply* msg)
    {
        auto distr = static_cast<const infcomp::VMF*>(msg->distribution());
        flatbuffers::Vector<double>::iterator mu_ptr = distr->proposal_mu()->data()->begin();
        auto dim = distr->proposal_mu()->data()->size();
        return vmf_distribution<RealType>(mu_ptr, mu_ptr + dim, distr->proposal_kappa());
    }
};

template<class RealType>
constexpr infcomp::Distribution proposal<boost::random::normal_distribution, RealType>::type_enum;

template<class RealType>
constexpr infcomp::Distribution proposal<vmf_distribution, RealType>::type_enum;

template<class IntType>
constexpr infcomp::Distribution proposal<boost::random::uniform_smallint, IntType>::type_enum;

}  // end namespace cpprob
#endif  // INCLUDE_TRAITS_HPP_
