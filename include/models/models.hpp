#ifndef INCLUDE_MODELS_HPP_
#define INCLUDE_MODELS_HPP_

#include <vector>
#include <utility>
#include <array>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/discrete_distribution.hpp>


#include "cpprob/cpprob.hpp"

namespace cpprob {
namespace models {

void gaussian_unknown_mean(const double y1, const double y2);

template<std::size_t N>
void linear_gaussian_1d (const std::array<double, N> & obs)
{
    using boost::random::normal_distribution;

    double state = 0;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        normal_distribution<> transition_distr {state, 1};
        state = cpprob::sample(transition_distr, true);
        cpprob::predict(state);
        normal_distribution<> likelihood {state, 1};
        cpprob::observe(likelihood, obs[i]);
    }
}

template<std::size_t N>
void hmm(const std::array<double, N> & observed_states)
{
    using boost::random::normal_distribution;
    using boost::random::discrete_distribution;
    using boost::random::uniform_smallint;

    constexpr int k = 3;
    static std::array<double, k> state_mean {-1, 0, 1};
    static std::array<std::array<double, k>, k> T {{{ 0.1,  0.5,  0.4 },
                                                    { 0.2,  0.2,  0.6 },
                                                    { 0.15, 0.15, 0.7 }}};
    uniform_smallint<> prior {0, 2};
    auto state = cpprob::sample(prior, true);
    cpprob::predict(state);
    for (const auto & obs_state : observed_states) {
        discrete_distribution<> transition_distr {T[state].begin(), T[state].end()};
        state = cpprob::sample(transition_distr, true);
        cpprob::predict(state);
        normal_distribution<> likelihood {state_mean[state], 1};
        cpprob::observe(likelihood, obs_state);
    }
}

void least_sqr(const std::vector<std::pair<double, double>> & points);
void all_distr();
} // end namespace models
} // end namespace cpprob

#endif  // INCLUDE_MODELS_HPP_
