#ifndef INCLUDE_MODELS_HPP_
#define INCLUDE_MODELS_HPP_

#include <vector>
#include <utility>

namespace cpprob {
namespace models {
void least_sqr(std::vector<std::pair<double, double>> points);
void mean_normal(const int y1);
void all_distr(const int y1, const int y2);
} // end namespace models
} // end namespace cpprob

#endif  // INCLUDE_MODELS_HPP_
