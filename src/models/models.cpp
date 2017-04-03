#include "models.hpp"

#include <algorithm>
#include <iterator>
#include <array>
#include <utility>

#include <boost/random/beta_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>

#include "cpprob.hpp"

/*
void f(cpprob::Core& c) {
    using boost::random::beta_distribution;
    using boost::random::bernoulli_distribution;

    static const beta_distribution<double> beta{1, 1};
    auto x = c.sample(beta);

    const bernoulli_distribution<double> ber{x};
    c.observe(ber, 1.0);
}
*/

void g(cpprob::Core& c) {
    using boost::random::normal_distribution;

    constexpr auto n = 6;
    static const normal_distribution<double> normal{0, 10};
    static const std::array<std::pair<double, double>, n> arr
                    = {{{1.0, 2.1},
                        {2.0, 3.9},
                        {3.0, 5.3},
                        {4.0, 7.7},
                        {5.0, 10.2},
                        {6.0, 12.9}}};

    const auto slope = c.sample(normal);
    const auto bias = c.sample(normal);
    for (size_t i = 0; i < n; ++i) {
        c.observe(
            normal_distribution<double>{slope*arr[i].first + bias, 1},
            arr[i].second);
    }
}

void mean_normal(cpprob::Core& c) {
    using boost::random::normal_distribution;

    static const normal_distribution<> normal{0, 1};
    const double y1 = 0.2, y2 = 0.2;
    auto mean = c.sample(normal);

    c.observe(normal_distribution<double>{mean, 1}, y1);
    c.observe(normal_distribution<double>{mean, 1}, y2);
}
