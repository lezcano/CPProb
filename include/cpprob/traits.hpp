#ifndef INCLUDE_TRAITS_HPP_
#define INCLUDE_TRAITS_HPP_

#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include "flatbuffers/infcomp_generated.h"

namespace cpprob {


template<class RealType = double>
boost::math::normal_distribution<RealType>
math_distr(const boost::random::normal_distribution<RealType>& d) {
    return boost::math::normal_distribution<RealType>{d.mean(), d.sigma()};
}

template<template <class ...> class Distr>
struct proposal {};

template<>
struct proposal<boost::random::normal_distribution> {
    //static const auto type_enum = infcomp::ProposalDistribution::ProposalDistribution_NormalProposal;
    static const infcomp::ProposalDistribution type_enum;

    template<class ...Params>
    static flatbuffers::Offset<void> request(flatbuffers::FlatBufferBuilder& fbb,
            const boost::random::normal_distribution<Params...>&) {
        return infcomp::CreateNormalProposal(fbb).Union();
    }

    template<class RealType = double>
    static boost::random::normal_distribution<RealType>
    get_distr(const infcomp::ProposalReply * msg) {
        auto param = static_cast<const infcomp::NormalProposal*>(msg->proposal());
        return boost::random::normal_distribution<RealType>(param->mean(), param->std());
    }
};


}  // end namespace cpprob
#endif  // INCLUDE_TRAITS_HPP_
