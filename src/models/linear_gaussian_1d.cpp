#include "models/models.hpp"
#include <vector>

#include <boost/random/normal_distribution.hpp>

#include "cpprob/cpprob.hpp"

namespace cpprob {
namespace models {

void linear_gaussian_1d (const std::vector<double> & obs)
{
    using boost::random::normal_distribution;

    double state = 0;
    for (std::size_t i = 0; i < obs.size(); ++i) {
        normal_distribution<> next_state_distr {state, 1};
        state = cpprob::sample(next_state_distr, true);
        cpprob::predict(state);
        normal_distribution<> obs_distr {state, 1};
        cpprob::observe(obs_distr, obs[i]);
    }
}

} // end namespace models
} // end namespace cpprob
