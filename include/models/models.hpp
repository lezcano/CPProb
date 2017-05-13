#ifndef INCLUDE_MODELS_HPP_
#define INCLUDE_MODELS_HPP_

#include <vector>
#include <utility>

namespace cpprob {
namespace models {

void gaussian_unknown_mean(const double y1, const double y2);
void linear_gaussian_1d (const std::vector<double> & obs);
void hmm(const std::vector<double> & observed_states);

void least_sqr(const std::vector<std::pair<double, double>> & points);
void mean_normal(const int y1);
void all_distr();
} // end namespace models
} // end namespace cpprob

#endif  // INCLUDE_MODELS_HPP_
