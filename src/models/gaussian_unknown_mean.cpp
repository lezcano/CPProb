#include "models/models.hpp"

#include <boost/random/normal_distribution.hpp>

#include "cpprob/cpprob.hpp"

namespace cpprob {
namespace models {

void gaussian_unknown_mean(const double y1, const double y2)
{
    using boost::random::normal_distribution;
    normal_distribution<> prior {1, 5};
    const double mu = cpprob::sample(prior, true);
    cpprob::predict(mu);
    const double var = 2;

    normal_distribution<> obs_distr {mu, var};
    cpprob::observe(obs_distr, y1);
    cpprob::observe(obs_distr, y2);
}

} // end namespace models
} // end namespace cpprob