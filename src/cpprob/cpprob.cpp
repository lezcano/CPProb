#include "cpprob/cpprob.hpp"


#include "cpprob/utils.hpp"
#include "cpprob/any.hpp"
#include "cpprob/state.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {
#include <iostream>

void predict(const cpprob::any & x, const std::string & addr)
{
    if (State::current_state() == StateType::inference ||
        State::current_state() == StateType::importance_sampling) {
        if (addr.empty()) {
            State::add_predict(get_addr(), x);
        }
        else {
            State::add_predict(addr, x);
        }
    }
}

void set_state(StateType s)
{
    State::set(s);
}

} // end namespace cpprob
