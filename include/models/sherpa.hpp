#ifndef CPPROB_SHERPA_HPP
#define CPPROB_SHERPA_HPP
#include <vector>

namespace sherpa_detail {

  void sherpa_wrapper(const std::vector<std::vector<std::vector<double>>> & observes);

  std::vector<std::vector<std::vector<double>>> sherpa();
}
#endif //CPPROB_SHERPA_HPP
