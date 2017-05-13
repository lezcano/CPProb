#include "models/models.hpp"

#include <array>
#include <vector>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/discrete_distribution.hpp>


#include "cpprob/cpprob.hpp"

namespace cpprob {
namespace models {

void hmm(const std::vector<double> & observed_states)
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
        discrete_distribution<> next_state_distr {T[state].begin(), T[state].end()};
        state = cpprob::sample(next_state_distr, true);
        cpprob::predict(state);
        normal_distribution<> obs_distr {state_mean[state], 1};
        cpprob::observe(obs_distr, obs_state);
    }
}

} // end namespace models
} // end namespace cpprob
