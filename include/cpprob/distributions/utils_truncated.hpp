#ifndef CPPROB_UTILS_TRUNCATED_HPP
#define CPPROB_UTILS_TRUNCATED_HPP

#include <cmath>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/truncated.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template <class Distribution>
struct logpdf<truncated<Distribution>> {
    auto operator()(const truncated<Distribution> & distr,
                    const typename truncated<Distribution>::result_type & x) -> decltype(logpdf<Distribution>()(distr.distribution(), x)) const
    {
        if (x < distr.min() || x > distr.max()) {
            return -std::numeric_limits<decltype(logpdf<Distribution>()(distr.distribution(), x))>::infinity();
        }
        const auto underlying_distr = distr.distribution();
        return logpdf<Distribution>()(underlying_distr, x) -
               std::log(normalise<Distribution>()(underlying_distr, distr.min(), distr.max()));
    }
};

template<class Distribution>
struct buffer<truncated<Distribution>> {
    using type = protocol::Truncated;
};

template<class Distribution>
struct from_flatbuffers<truncated<Distribution>> {
    using distr_t = truncated<Distribution>;

    distr_t operator()(const buffer_t<distr_t> * distr_fbb)
    {
        const auto inner_distr_fbb = distr_fbb->distribution()->template distribution_as<buffer_t<Distribution>>();
        Distribution distr = from_flatbuffers<Distribution>()(inner_distr_fbb);
        return truncated<Distribution>(distr, distr_fbb->min(), distr_fbb->max());
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_TRUNCATED_HPP
