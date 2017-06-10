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
    normal_distribution<RealType> prior {1, 5};
    const RealType mu = cpprob::sample(prior, true);
    const RealType var = 2;

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
