#ifndef CPPROB_UTILS_UNIFORM_REAL_HPP
#define CPPROB_UTILS_UNIFORM_REAL_HPP

#include <cmath>

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include "cpprob/distributions/utils_base.hpp"
#include "cpprob/distributions/mixture.hpp"
#include "cpprob/distributions/truncated.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////

template<typename RealType>
struct logpdf<boost::random::uniform_real_distribution<RealType>> {
    RealType operator()(const boost::random::uniform_real_distribution<RealType>& distr,
                        const typename boost::random::uniform_real_distribution<RealType>::result_type &) const
    {
        return -std::log(distr.b()-distr.a());
    }
};

//////////////////////////////
/////////// Prior  ///////////
//////////////////////////////

template<class RealType>
struct proposal<boost::random::uniform_real_distribution<RealType>> {
    using type = mixture<truncated<boost::random::normal_distribution<RealType>>>;
};

template<class RealType>
struct buffer<boost::random::uniform_real_distribution<RealType>> {
    using type = infcomp::protocol::UniformContinuousAlt;
};

template<class RealType>
struct serialise<boost::random::uniform_real_distribution<RealType>> {
    using prior = boost::random::uniform_real_distribution<RealType>;

    static proposal_t<prior> from_flatbuffers(const infcomp::protocol::ProposalReply *msg) {
        auto distr = static_cast<const buffer_t<prior>*>(msg->distribution());

        // We suppose that hte three vectors have the same length
        auto num_components = distr->proposal_coeffs()->data()->size();

        auto coef_ptr = distr->proposal_coeffs()->data()->begin();
        const auto coef = std::vector<RealType>(coef_ptr, coef_ptr + num_components);

        auto means_ptr = distr->proposal_means()->data()->begin();
        auto std_ptr = distr->proposal_stds()->data()->begin();

        const auto min = distr->prior_min();
        const auto max = distr->prior_max();

        std::vector<truncated<boost::random::normal_distribution<RealType>>> v_distr(num_components);
        for (std::size_t i = 0; i < num_components; ++i) {
            v_distr[i] = truncated<boost::normal_distribution<RealType>>(
                    boost::normal_distribution<RealType>(*means_ptr, *std_ptr),
                            min, max);
            ++means_ptr;
            ++std_ptr;
        }

        return proposal_t<prior>(coef.begin(), coef.end(), v_distr.begin(), v_distr.end());
    }

    static flatbuffers::Offset<void> to_flatbuffers(flatbuffers::FlatBufferBuilder& buff, const prior & distr) {
        return infcomp::protocol::CreateUniformContinuous(buff, distr.a(), distr.b()).Union();
    }
};

} // end namespace cpprob

#endif //CPPROB_UTILS_UNIFORM_REAL_HPP
