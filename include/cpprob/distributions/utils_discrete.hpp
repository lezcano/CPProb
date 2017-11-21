#ifndef CPPROB_UTILS_DISCRETE_HPP
#define CPPROB_UTILS_DISCRETE_HPP

#include <cmath>

#include <boost/random/discrete_distribution.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
////// Prior & Proposal //////
//////////////////////////////

template<class IntType, class WeightType>
struct logpdf <boost::random::discrete_distribution<IntType, WeightType>>{
    WeightType operator()(const boost::random::discrete_distribution<IntType, WeightType> &distr,
                          const typename boost::random::discrete_distribution<IntType, WeightType>::result_type &x) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<WeightType>::infinity();
        }
        return std::log(distr.probabilities()[static_cast<std::size_t>(x)]);
    }
};

template<class IntType, class WeightType>
struct buffer<boost::random::discrete_distribution<IntType, WeightType>> {
    using type = protocol::Discrete;
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class IntType, class WeightType>
struct proposal<boost::random::discrete_distribution<IntType, WeightType>> {
    using type = boost::random::discrete_distribution<IntType, WeightType>;
};

template<class IntType, class WeightType>
struct to_flatbuffers<boost::random::discrete_distribution<IntType, WeightType>> {
    using distr_t = boost::random::discrete_distribution<IntType, WeightType>;

    flatbuffers::Offset<void> operator()(flatbuffers::FlatBufferBuilder& buff,
                                                    const distr_t & distr,
                                                    typename distr_t::result_type value)
    {
        return protocol::CreateDiscrete(buff, 0, buff.CreateVector<double>(distr.probabilities()), value).Union();
    }
};

//////////////////////////////
///////// Proposal  //////////
//////////////////////////////

template<class IntType, class WeightType>
struct from_flatbuffers<boost::random::discrete_distribution<IntType, WeightType>> {
    using distr_t = boost::random::discrete_distribution<IntType, WeightType>;

    distr_t operator()(const buffer_t<distr_t> * distr)
    {
        return distr_t(distr->probabilities()->begin(), distr->probabilities()->end());
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_DISCRETE_HPP
