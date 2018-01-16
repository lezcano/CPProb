#include "cpprob/cpprob.hpp"
#include "cpprob/state.hpp"

namespace cpprob {

void start_rejection_sampling()
{
    State::start_rejection_sampling();
}

void finish_rejection_sampling()
{
    State::finish_rejection_sampling();
}

rejection_sampling::rejection_sampling() { start_rejection_sampling(); }

rejection_sampling::~rejection_sampling() { finish_rejection_sampling(); }

} // end namespace cpprob
