#ifndef INCLUDE_TRAITS_HPP_
#define INCLUDE_TRAITS_HPP_

#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <boost/random/beta_distribution.hpp>
#include <boost/math/distributions/beta.hpp>


#include <boost/random/bernoulli_distribution.hpp>
#include <boost/math/distributions/bernoulli.hpp>

namespace cpprob {

template<class RealType = double>
boost::math::normal_distribution<RealType>
    math_distr(const boost::random::normal_distribution<RealType>& d) {
    return boost::math::normal_distribution<RealType>{d.mean(), d.sigma()};
}

template<class RealType = double>
boost::math::bernoulli_distribution<RealType>
    math_distr(const boost::random::bernoulli_distribution<RealType>& d) {
    return boost::math::bernoulli_distribution<RealType>{d.p()};
}

template<class RealType = double>
boost::math::beta_distribution<RealType>
    math_distr(const boost::random::beta_distribution<RealType>& d) {
    return boost::math::beta_distribution<RealType>{d.alpha(), d.beta()};
}

template<template <class...> class Distr, class... Params>
struct poposal_distr {};

template<class...Params>
struct poposal_distr<boost::random::normal_distribution, Params ...>{
    using type = boost::random::normal_distribution<Params ...>;
};

template<template <class...> class Distr, class... Params>
using poposal_distr_t = typename poposal_distr<Distr, Params ...>::type;


}  // end namespace cpprob
#endif  // INCLUDE_TRAITS_HPP_
