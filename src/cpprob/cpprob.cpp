#include "cpprob/cpprob.hpp"

#include "cpprob/utils.hpp"
#include "cpprob/state.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {

void predict(const NDArray<double> &x) {
    if (!State::training) {
        auto addr = get_addr();
        State::add_predict(addr, x);
    }
}

} // end namespace cpprob
