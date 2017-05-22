#include "cpprob/cpprob.hpp"

#include "cpprob/utils.hpp"
#include "cpprob/state.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {
#include <iostream>

void predict(const NDArray<double> &x) {
    if (State::current_state() == StateType::inference ||
        State::current_state() == StateType::importance_sampling) {
        auto addr = get_addr();
        State::add_predict(addr, x);
    }
}

void set_state(StateType s)
{
    State::set(s);
}

} // end namespace cpprob
