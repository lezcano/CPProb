#ifndef CPPROB_SHERPA_MINI_HPP
#define CPPROB_SHERPA_MINI_HPP

#include <vector>

namespace cpprob {
namespace models {

void sherpa_wrapper(const std::vector<double>& test_image);
std::vector<double> dummy_sherpa();

} // end namespace models
} // end namespace cpprob
#endif //CPPROB_SHERPA_MINI_HPP
