#ifndef CPPROB_DISTRIBUTION_UTILS_HPP
#define CPPROB_DISTRIBUTION_UTILS_HPP

#include <numeric>
#include <functional>
#include <limits>
#include <cmath>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/discrete_distribution.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/beta.hpp>

#include "min_max_discrete.hpp"
#include "min_max_continuous.hpp"
#include "multivariate_normal.hpp"

#include "cpprob/ndarray.hpp"

#include "flatbuffers/infcomp_generated.h"

namespace cpprob {
template <class RealType>
RealType logpdf(const boost::random::normal_distribution<RealType>& distr,
                const typename boost::random::normal_distribution<RealType>::result_type & x)
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

template <class IntType>
double logpdf(const boost::random::uniform_smallint<IntType>& distr,
              const typename boost::random::uniform_smallint<IntType>::result_type &)
{
    return -std::log(distr.max() - distr.min() + 1.0);
}

template<class IntType, class WeightType>
WeightType logpdf(const min_max_discrete_distribution<IntType, WeightType>& distr,
                  const typename min_max_discrete_distribution<IntType, WeightType>::result_type & x)
{
    if (x < distr.min() || x > distr.max()) {
        return -std::numeric_limits<WeightType>::infinity();
    }
    return std::log(distr.probabilities()[static_cast<std::size_t>(x-distr.min())]);
}

template<class IntType, class WeightType>
WeightType logpdf(const boost::random::discrete_distribution<IntType, WeightType>& distr,
                  const typename boost::random::discrete_distribution<IntType, WeightType>::result_type & x)
{
    if (x < distr.min() || x > distr.max()) {
        return -std::numeric_limits<WeightType>::infinity();
    }
    return std::log(distr.probabilities()[static_cast<std::size_t>(x)]);
}

template<class RealType>
RealType logpdf(const min_max_continuous_distribution<RealType>& distr,
                const typename min_max_continuous_distribution<RealType>::result_type & x) {
    auto beta_distr = distr.beta();
    auto a = beta_distr.alpha();
    auto b = beta_distr.beta();
    auto min = distr.min();
    auto max = distr.max();
    auto norm_x = (x - min) / (max - min);
    if (norm_x <= 0 || norm_x >= 1) {
        return -std::numeric_limits<RealType>::infinity();
    }

    return std::log(boost::math::pdf(boost::math::beta_distribution<RealType>(a, b), norm_x))
           - std::log(max - min);
}

template<typename IntType, class RealType>
RealType logpdf(const boost::random::poisson_distribution<IntType, RealType>& distr,
                const typename boost::random::poisson_distribution<IntType, RealType>::result_type & x)
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

template<typename RealType>
RealType logpdf(const boost::random::uniform_real_distribution<RealType>& distr,
                const typename boost::random::uniform_real_distribution<RealType>::result_type &)
{
    return -std::log(distr.b()-distr.a());
}

template<typename RealType>
RealType logpdf(const multivariate_normal_distribution<RealType>& distr,
                const typename multivariate_normal_distribution<RealType>::result_type & x)
{
    RealType ret = 0;
    auto vec_distr = distr.distr();
    auto it_distr = vec_distr.begin();
    auto it_x = x.begin();
    for(; it_distr != vec_distr.end() && it_x != x.end(); ++it_distr, ++it_x)
        ret += logpdf(*it_distr, *it_x);
    return ret;
}


template<template <class ...> class Distr, class ...Params>
struct proposal;

template<class RealType>
struct proposal<boost::random::normal_distribution, RealType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::Normal;

    using type = boost::random::normal_distribution<RealType>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto param = static_cast<const infcomp::protocol::Normal*>(msg->distribution());
        return boost::random::normal_distribution<RealType>(param->proposal_mean(), param->proposal_std());
    }
};


template<class IntType>
struct proposal<boost::random::uniform_smallint, IntType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::UniformDiscrete;

    using type = min_max_discrete_distribution<IntType, double>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto param = static_cast<const infcomp::protocol::UniformDiscrete*>(msg->distribution());
        flatbuffers::Vector<double>::iterator probs_ptr = param->proposal_probabilities()->data()->begin();
        return min_max_discrete_distribution<IntType, double>(param->prior_min(),
                                                              param->prior_min() + param->prior_size() - 1,
                                                              probs_ptr, probs_ptr+param->prior_size());
    }
};

template<class IntType, class WeightType>
struct proposal<boost::random::discrete_distribution, IntType, WeightType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::Discrete;

    using type = boost::random::discrete_distribution<IntType, WeightType>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto param = static_cast<const infcomp::protocol::Discrete*>(msg->distribution());
        flatbuffers::Vector<double>::iterator probs_ptr = param->proposal_probabilities()->data()->begin();
        return boost::random::discrete_distribution<IntType, WeightType>(probs_ptr, probs_ptr+param->prior_size());
    }
};

template<class RealType>
struct proposal<boost::random::uniform_real_distribution, RealType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::UniformContinuous;

    using type = min_max_continuous_distribution<RealType>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto distr = static_cast<const infcomp::protocol::UniformContinuous*>(msg->distribution());
        return min_max_continuous_distribution<RealType>(distr->prior_min(), distr->prior_max(),
                                                         distr->proposal_mode(), distr->proposal_certainty());
    }
};

template<class IntType, class RealType>
struct proposal<boost::random::poisson_distribution, IntType, RealType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::Poisson;

    using type = boost::random::poisson_distribution<IntType, RealType>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto distr = static_cast<const infcomp::protocol::Poisson*>(msg->distribution());
        return boost::random::poisson_distribution<IntType, RealType>(distr->proposal_lambda());
    }
};


template<class RealType>
struct proposal<multivariate_normal_distribution, RealType> {
    static constexpr auto type_enum = infcomp::protocol::Distribution::MultivariateNormal;

    using type = multivariate_normal_distribution<RealType>;

    static type get_distr(const infcomp::protocol::ProposalReply* msg)
    {
        auto distr = static_cast<const infcomp::protocol::MultivariateNormal*>(msg->distribution());
        auto mean_ptr = distr->proposal_mean()->data()->begin();
        auto shape_ptr = distr->proposal_mean()->shape()->begin();
        auto sigma_ptr = distr->proposal_sigma()->data()->begin();
        auto dim = distr->proposal_mean()->data()->size();
        auto shape_size = distr->proposal_mean()->shape()->size();

        auto vec_data = std::vector<RealType>(mean_ptr, mean_ptr+dim);
        auto vec_shape = std::vector<int>(shape_ptr, shape_ptr+shape_size);
        return multivariate_normal_distribution<RealType>(NDArray<RealType>(std::move(vec_data), std::move(vec_shape)),
                                                          sigma_ptr, sigma_ptr + dim);
    }
};

template<class RealType>
constexpr infcomp::protocol::Distribution proposal<boost::random::normal_distribution, RealType>::type_enum;

template<class IntType>
constexpr infcomp::protocol::Distribution proposal<boost::random::uniform_smallint, IntType>::type_enum;

template<class RealType>
constexpr infcomp::protocol::Distribution proposal<boost::random::uniform_real_distribution, RealType>::type_enum;

template<class IntType, class RealType>
constexpr infcomp::protocol::Distribution proposal<boost::random::poisson_distribution, IntType, RealType>::type_enum;

template<class RealType>
constexpr infcomp::protocol::Distribution proposal<multivariate_normal_distribution, RealType>::type_enum;

template<class IntType, class WeightType>
constexpr infcomp::protocol::Distribution proposal<boost::random::discrete_distribution, IntType, WeightType>::type_enum;

}  // end namespace cpprob
#endif //CPPROB_DISTRIBUTION_UTILS_HPP