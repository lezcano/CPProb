#ifndef CPPROB_UTILS_UNIFORM_REAL_HPP
#define CPPROB_UTILS_UNIFORM_REAL_HPP

#include <cmath>
#include <limits>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/mixture.hpp"
#include "cpprob/distributions/truncated.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<typename RealType>
struct logpdf<boost::random::uniform_real_distribution<RealType>> {
    RealType operator()(const boost::random::uniform_real_distribution<RealType>& distr,
                        const typename boost::random::uniform_real_distribution<RealType>::result_type & x) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<RealType>::infinity();
        }
        return -std::log(distr.b()-distr.a());
    }
};

template<class RealType>
struct buffer<boost::random::uniform_real_distribution<RealType>> {
    using type = protocol::UniformContinuous;
};

template<class RealType>
struct proposal<boost::random::uniform_real_distribution<RealType>> {
    using type = truncated<mixture<boost::random::normal_distribution<RealType>, RealType>>;
};

template<class RealType>
struct to_flatbuffers<boost::random::uniform_real_distribution<RealType>> {
    using distr_t = boost::random::uniform_real_distribution<RealType>;

    flatbuffers::Offset<void> operator()(flatbuffers::FlatBufferBuilder& buff,
                                                    const distr_t & distr,
                                                    const typename distr_t::result_type value)
    {
        return protocol::CreateUniformContinuous(buff, distr.a(), distr.b(), value).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_REAL_HPP
