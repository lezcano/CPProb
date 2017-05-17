#include "models/models.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/distributions/vmf.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"

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

//-m infer -n 100 -o [(1 2.1) (2 3.9) (3 5.3) (4 7.7) (5 10.2) (6 12.9)]
void least_sqr(const std::vector<std::pair<double, double>>& points) {

}
template <std::size_t N>
void least_sqr(const std::array<std::pair<double, double>, N>& points) {
    using boost::random::normal_distribution;

    static normal_distribution<double> normal{0, 10};

    const auto slope = cpprob::sample(normal, true);
    const auto bias = cpprob::sample(normal, true);
    cpprob::predict(slope);
    cpprob::predict(bias);

    for (const auto& point : points) {
        auto obs_distr = normal_distribution<double>{slope * point.first + bias, 1};
        cpprob::observe(obs_distr, point.second);
    }
}

void all_distr() {
    using cpprob::sample;
    using cpprob::predict;
    using cpprob::observe;

    using boost::random::normal_distribution;
    normal_distribution<> normal{1,2};
    auto normal_val = sample(normal, true);
    predict(normal_val);
    observe(normal, normal_val);

    using boost::random::uniform_smallint;
    uniform_smallint<> discrete {2, 7};
    auto discrete_val = sample(discrete, true);
    predict(discrete_val);
    observe(discrete, discrete_val);

    using boost::random::uniform_real_distribution;
    uniform_real_distribution<> rand_unif {2, 9.5};
    auto rand_unif_val = sample(rand_unif, true);
    predict(rand_unif_val);
    observe(rand_unif, rand_unif_val);

    using boost::random::poisson_distribution;
    poisson_distribution<> poiss(0.8);
    auto poiss_val = sample(poiss, true);
    predict(poiss_val);
    observe(poiss, poiss_val);

    using cpprob::vmf_distribution;
    vmf_distribution<> vmf{{1,2,3}, 3};
    auto vmf_val = sample(vmf, true);
    predict(vmf_val);
    observe(vmf, vmf_val);

    using cpprob::multivariate_normal_distribution;
    multivariate_normal_distribution<> multi{{1,2,3,4}, {2,1,5,3}};
    auto sample_multi = sample(multi, true);
    predict(sample_multi);
    observe(multi, sample_multi);
}

} // end namespace models
} // end namespace cpprob
