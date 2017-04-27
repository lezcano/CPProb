#include "models/models.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/distr/vmf.hpp"

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

namespace cpprob {
namespace models {

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
    cpprob::predict(slope);
    cpprob::predict(bias);

    for (size_t i = 0; i < n; ++i) {
        auto obs_distr = normal_distribution<double>{slope * arr[i].first + bias, 1};
        cpprob::observe(obs_distr, arr[i].second);
    }
}

void mean_normal(const int y1) {
    using boost::random::normal_distribution;

    static normal_distribution<> normal{0, 1};

    auto mean = cpprob::sample(normal);
    cpprob::predict(mean);

    auto obs_distr = normal_distribution<double>{mean, 0.1};

    cpprob::observe(obs_distr, y1);
}

void all_distr(const int y1, const int y2) {
    using boost::random::normal_distribution;
    using boost::random::uniform_smallint;
    using boost::random::uniform_real_distribution;
    using boost::random::poisson_distribution;

    using cpprob::vmf_distribution;
    uniform_smallint<> discrete {2, 7};
    auto dis = cpprob::sample(discrete);
    cpprob::predict(dis);
    normal_distribution<> normal1{0,static_cast<double>(dis)};
    normal_distribution<> normal2{cpprob::sample(normal1),static_cast<double>(cpprob::sample(discrete))};
    vmf_distribution<> vmf({1,2,3,4}, 3);

    auto poiss = poisson_distribution<>(0.8);
    auto poiss_val = cpprob::sample(poiss);
    cpprob::predict(poiss_val);

    auto rand_unif = uniform_real_distribution<>(2, 9.5);
    auto rand_unif_val = cpprob::sample(rand_unif);
    cpprob::predict(rand_unif_val);

    auto x = cpprob::sample(vmf);
    cpprob::predict(x);

    cpprob::observe(rand_unif, rand_unif_val);
    cpprob::observe(vmf, x);
    cpprob::observe(normal2, y1);
    cpprob::observe(discrete, y2);
}

} // end namespace models
} // end namespace cpprob
