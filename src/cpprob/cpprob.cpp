#include "cpprob/cpprob.hpp"
#include "cpprob/state.hpp"
#include "cpprob/utils.hpp"

namespace cpprob {

void start_rejection_sampling()
{
    State::start_rejection_sampling();
}

void finish_rejection_sampling()
{
    State::finish_rejection_sampling();
}

void seed_sample_rng(decltype(get_sample_rng()()) seed)
{
    get_sample_rng().seed(seed);
}

void seed_observe_rng(decltype(get_observe_rng()()) seed)
{
    get_observe_rng().seed(seed);
}

} // end namespace cpprob
