#ifndef INCLUDE_TRAITS_HPP_
#define INCLUDE_TRAITS_HPP_

#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

namespace cpprob {

template<class RealType = double>
boost::math::normal_distribution<RealType>
    math_distr(const boost::random::normal_distribution<RealType>& d) {
    return boost::math::normal_distribution<RealType>{d.mean(), d.sigma()};
}


template<template <class ...> class Distr>
struct distr_name {};

template<>
struct distr_name<boost::random::normal_distribution> {
    static constexpr char value[] = "normal";
};

template<class RealType = double>
boost::random::normal_distribution<RealType>
    posterior_distr(const std::vector<RealType>& param) {
    return boost::random::normal_distribution<RealType>{param[0], param[1]};
}

}  // end namespace cpprob
#endif  // INCLUDE_TRAITS_HPP_
