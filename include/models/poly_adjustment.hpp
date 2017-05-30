#ifndef CPPROB_POLY_ADJUSTMENT_HPP
#define CPPROB_POLY_ADJUSTMENT_HPP

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <boost/random/normal_distribution.hpp>

#include "cpprob/cpprob.hpp"

template<class RealType>
std::vector<RealType> generate_polynomial(std::size_t degree)
{
    boost::random::normal_distribution<RealType> prior{0, 10};
    std::vector<RealType> ret(degree+1);
    for (std::size_t i = 0; i <= degree; ++i) {
        ret[i] = cpprob::sample(prior);
    }
    return ret;
}
template<class RealType>
RealType eval_poly(const std::vector<RealType> & poly, RealType point)
{
    // Horner's algorithm
    return std::accumulate(poly.crbegin(), poly.crend(), 0.0,
                           [point](RealType acc, RealType next) { return acc * point + next; });
}

//-m infer -n 100 -o [(1 2.1) (2 3.9) (3 5.3) (4 7.7) (5 10.2) (6 12.9)]
template <std::size_t D, std::size_t N, class RealType = double>
void poly_adjustment(const std::array<std::array<RealType, 2>, N>& points)
{
    auto poly = generate_polynomial<RealType>(D);

    for (const auto & point : points) {
        boost::random::normal_distribution<RealType> likelihood {eval_poly(poly, point[0]), 1};
        cpprob::observe(likelihood, point[1]);
    }
    for (std::size_t i = 0; i < poly.size(); ++i) {
        cpprob::predict(poly[i], "Coefficient " + std::to_string(i));
    }
}

#endif //CPPROB_POLY_ADJUSTMENT_HPP
