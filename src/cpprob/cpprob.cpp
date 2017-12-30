#include "cpprob/cpprob.hpp"
#include "cpprob/state.hpp"

namespace cpprob {

rejection_sampling::rejection_sampling() { State::start_rejection_sampling(); }

rejection_sampling::~rejection_sampling() { State::finish_rejection_sampling(); }

} // end namespace cpprob
