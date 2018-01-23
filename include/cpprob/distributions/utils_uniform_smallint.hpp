#ifndef CPPROB_UTILS_UNIFORM_SMALLINT_HPP
#define CPPROB_UTILS_UNIFORM_SMALLINT_HPP

#include <cmath>

#include <boost/random/uniform_smallint.hpp>

#include "flatbuffers/infcomp_generated.h"
#include "cpprob/distributions/utils_base.hpp"

namespace cpprob {

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template <class IntType>
struct logpdf<boost::random::uniform_smallint<IntType>> {
    double operator()(const boost::random::uniform_smallint<IntType>& distr,
                      const typename boost::random::uniform_smallint<IntType>::result_type & x) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<double>::infinity();
        }
        return -std::log(distr.max() - distr.min() + 1.0);
    }
};

template<class IntType>
struct buffer<boost::random::uniform_smallint<IntType>> {
    using type = protocol::UniformDiscrete;
};

template<class IntType>
struct proposal<boost::random::uniform_smallint<IntType>> {
    using type = min_max_discrete_distribution<IntType, double>;
};

template<class IntType>
struct to_flatbuffers<boost::random::uniform_smallint<IntType>> {
    using distr_t = boost::random::uniform_smallint<IntType>;

    flatbuffers::Offset<void> operator()(flatbuffers::FlatBufferBuilder & buff,
                                         const distr_t & distr,
                                         const typename distr_t::result_type value)
    {
        return protocol::CreateUniformDiscrete(buff,
                                               static_cast<int>(distr.a()),
                                               static_cast<int>(distr.b()),
                                               static_cast<int>(value)).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_SMALLINT_HPP
