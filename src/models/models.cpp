#include "models.hpp"

#include <boost/random/beta_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>

#include "cpprob.hpp"

/*
void f() {
    using boost::random::beta_distribution;
    using boost::random::bernoulli_distribution;

    static beta_distribution<double> beta{1, 1};
    auto x = sample(beta);

    bernoulli_distribution<double> ber{x};
    observe(ber, 1.0);
}
*/

void g() {
    using boost::random::normal_distribution;

    constexpr auto n = 6;
    static normal_distribution<double> normal{0, 10};
    static const std::array<std::pair<double, double>, n> arr
                    = {{{1.0, 2.1},
                        {2.0, 3.9},
                        {3.0, 5.3},
                        {4.0, 7.7},
                        {5.0, 10.2},
                        {6.0, 12.9}}};

    const auto slope = cpprob::sample(normal);
    const auto bias = cpprob::sample(normal);
    for (size_t i = 0; i < n; ++i) {
        auto obs_distr = normal_distribution<double>{slope*arr[i].first + bias, 1};
        cpprob::observe(obs_distr, arr[i].second);
    }
}

void mean_normal() {
    using boost::random::normal_distribution;

    static normal_distribution<> normal{0, 1};
    const double y1 = 0.2, y2 = 0.2;
    auto mean = cpprob::sample(normal);

    auto obs_distr = normal_distribution<double>{mean, 1};

    cpprob::observe(obs_distr, y1);
    cpprob::observe(obs_distr, y2);
}
