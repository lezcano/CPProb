// File: "models/gaussian.cpp"
#include <boost/random/normal_distribution.hpp>
#include "cpprob/cpprob.hpp"

namespace models {
void gaussian_unknown_mean(const double x1, const double x2)
{
    constexpr double mu0 = 1, sigma0 = 1.5, sigma = 2;      // Hyperparameters

    boost::normal_distribution<> prior {mu0, sigma0};
    const double mu = cpprob::sample(prior, true);
    boost::normal_distribution<> likelihood {mu, sigma};

    cpprob::observe(likelihood, x1);
    cpprob::observe(likelihood, x2);
    cpprob::predict(mu, "Mean");
}
}
