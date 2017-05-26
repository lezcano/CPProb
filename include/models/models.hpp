#ifndef INCLUDE_MODELS_HPP_
#define INCLUDE_MODELS_HPP_

#include <vector>
#include <utility>
#include <array>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/math/distributions/normal.hpp>


#include "cpprob/cpprob.hpp"

namespace models {

template<class RealType>
void gaussian_unknown_mean(const RealType y1, const RealType y2)
{
    using boost::random::normal_distribution;
    normal_distribution<RealType> prior {1, 5};
    const RealType mu = cpprob::sample(prior, true);
    cpprob::predict(mu);
    const RealType var = 2;

    normal_distribution<RealType> obs_distr {mu, var};
    cpprob::observe(obs_distr, y1);
    cpprob::observe(obs_distr, y2);
}

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

//-m infer -n 100 -o [(1 2.1) (2 3.9) (3 5.3) (4 7.7) (5 10.2) (6 12.9)]
template <std::size_t N>
void least_sqr(const std::array<std::pair<double, double>, N>& points)
{
    using boost::random::normal_distribution;

    static normal_distribution<double> normal{0, 10};

    const auto slope = cpprob::sample(normal, true);
    const auto bias = cpprob::sample(normal, true);
    cpprob::predict(slope);
    cpprob::predict(bias);

    for (const auto& point : points) {
        auto likelihood = normal_distribution<double>{slope * point.first + bias, 1};
        cpprob::observe(likelihood, point.second);
    }
}

template<class RealType=double>
void normal_rejection_sampling(const RealType y1, const RealType y2)
{
    using boost::random::normal_distribution;
    using boost::random::uniform_real_distribution;
    const RealType sigma = 0.01;
    const int max_it = 10000;
    int it = 0;

    // Sample from Normal Distr
    const auto maxval = boost::math::pdf(boost::math::normal_distribution<RealType>(1, sigma), 1);
    uniform_real_distribution<RealType> proposal {-5*sigma, 5*sigma};
    uniform_real_distribution<RealType> accept {0, 1};
    RealType mu;


    while(it < max_it) {
        auto p = cpprob::sample(proposal, true);
        mu = cpprob::sample(accept, true);
        if (p < boost::math::pdf(boost::math::normal_distribution<RealType>(1, sigma), mu)/maxval) {
            break;
        }
        ++it;
    }

    const RealType var = std::sqrt(2);
    normal_distribution<RealType> obs_distr {mu, var};
    cpprob::observe(obs_distr, y1);
    cpprob::observe(obs_distr, y2);
    cpprob::predict(mu);
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
#endif  // INCLUDE_MODELS_HPP_
