#ifndef INCLUDE_CPPROB_HPP_
#define INCLUDE_CPPROB_HPP_

#include <vector>

#include "cpprob/impl/cpprob.hpp"
#include "cpprob/trace.hpp"

namespace cpprob {

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, double x);

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr);

template<class Func>
void compile(const Func& f, const std::string& tcp_addr);

template<class Func, class... Args>
std::vector<std::vector<double>> inference(
        const Func& f,
        const std::string& tcp_addr,
        size_t n,
        Args&&... args);
}  // namespace cpprob
#endif  // INCLUDE_CPPROB_HPP_
