#ifndef INCLUDE_MODELS_HPP_
#define INCLUDE_MODELS_HPP_

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/math/distributions/normal.hpp>


#include "cpprob/cpprob.hpp"

namespace models {
template<class RealType = double>
void gaussian_unknown_mean(const RealType y1, const RealType y2)
{
    using boost::random::normal_distribution;
    normal_distribution<RealType> prior {1, std::sqrt(5)};
    const RealType mu = cpprob::sample(prior, true);
    const RealType var = std::sqrt(2);

    normal_distribution<RealType> obs_distr {mu, var};
    cpprob::observe(obs_distr, y1);
    cpprob::observe(obs_distr, y2);
    cpprob::predict(mu, "Mu");
}

template<std::size_t N>
void linear_gaussian_1d (const std::array<double, N> & observations)
{
    using boost::random::normal_distribution;

    double state = 0;
    for (const auto obs : observations) {
        normal_distribution<> transition_distr {state, 1};
        state = cpprob::sample(transition_distr, true);
        normal_distribution<> likelihood {state, 1};
        cpprob::observe(likelihood, obs);
        cpprob::predict(state, "State");
    }
}

template<class RealType=double>
void normal_rejection_sampling(const RealType y1, const RealType y2)
{
    using boost::random::normal_distribution;
    using boost::random::uniform_real_distribution;

    const RealType mu_prior = 1;
    const RealType sigma_prior = std::sqrt(5);
    const RealType sigma = std::sqrt(2);

    const int max_it = 10000;
    int it = 0;

    // Sample from Normal Distr
    const auto maxval = boost::math::pdf(boost::math::normal_distribution<RealType>(mu_prior, sigma_prior), mu_prior);
    uniform_real_distribution<RealType> proposal {-5*sigma_prior, 5*sigma_prior};
    uniform_real_distribution<RealType> accept {0, 1};
    RealType mu;


    while(it < max_it) {
        auto p = cpprob::sample(proposal, true);
        mu = cpprob::sample(accept, true);
        if (p < boost::math::pdf(boost::math::normal_distribution<RealType>(mu_prior, sigma_prior), mu)/maxval) {
            break;
        }
        ++it;
    }

    normal_distribution<RealType> likelihood {mu, sigma};
    cpprob::observe(likelihood, y1);
    cpprob::observe(likelihood, y2);
    cpprob::predict(mu, "Mu");
}
/*
    Truth
   [[ 0.3775 0.3092 0.3133]
    [ 0.0416 0.4045 0.5539]
    [ 0.0541 0.2552 0.6907]
    [ 0.0455 0.2301 0.7244]
    [ 0.1062 0.1217 0.7721]
    [ 0.0714 0.1732 0.7554]
    [ 0.9300 0.0001 0.0699]
    [ 0.4577 0.0452 0.4971]
    [ 0.0926 0.2169 0.6905]
    [ 0.1014 0.1359 0.7626]
    [ 0.0985 0.1575 0.744 ]
    [ 0.1781 0.2198 0.6022]
    [ 0.0000 0.9848 0.0152]
    [ 0.1130 0.1674 0.7195]
    [ 0.0557 0.1848 0.7595]
    [ 0.2017 0.0472 0.7511]
    [ 0.2545 0.0611 0.6844]]
*/

template<std::size_t N>
void hmm(const std::array<double, N> & observed_states)
{
    using boost::random::normal_distribution;
    using boost::random::discrete_distribution;
    using boost::random::uniform_smallint;

    constexpr int k = 3;
    static std::array<double, k> state_mean {{-1, 0, 1}};
    static std::array<std::array<double, k>, k> T {{{{ 0.1,  0.5,  0.4 }},
                                                    {{ 0.2,  0.2,  0.6 }},
                                                    {{ 0.15, 0.15, 0.7 }}}};
    uniform_smallint<> prior {0, 2};
    auto state = cpprob::sample(prior, true);
    cpprob::predict(state, "Initial state");
    for (const auto obs : observed_states) {
        discrete_distribution<> transition_distr {T[state].begin(), T[state].end()};
        state = cpprob::sample(transition_distr, true);
        cpprob::predict(state, "State");
        normal_distribution<> likelihood {state_mean[state], 1};
        cpprob::observe(likelihood, obs);
    }
}

void all_distr();
} // end namespace models
#endif  // INCLUDE_MODELS_HPP_
