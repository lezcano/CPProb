#ifndef CPPROB_POLY_ADJUSTMENT_HPP
#define CPPROB_POLY_ADJUSTMENT_HPP

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <boost/random/normal_distribution.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/metapriors.hpp"

namespace models {

template<class RealType>
std::vector<RealType> generate_polynomial(std::size_t degree) {
    boost::random::normal_distribution<RealType> prior{0, 10};
    std::vector<RealType> ret(degree + 1);
    for (auto& coef : ret) {
        coef = cpprob::sample(prior, true);
    }
    return ret;
}

template<class RealType>
RealType eval_poly(const std::vector<RealType> &poly, RealType point) {
    // Horner's algorithm
    return std::accumulate(poly.crbegin(), poly.crend(), 0.0,
                           [point](RealType acc, RealType next) { return acc * point + next; });
}

//-m infer -n 100 -o [[1 2.1] [2 3.9] [3 5.3] [4 7.7] [5 10.2] [6 12.9]]
template<std::size_t D, class RealType = double>
void poly_adjustment_prior(const std::vector<std::pair<RealType, RealType>> &points,
                    cpprob::Builder<
                        cpprob::Prior<
                            std::vector<
                                std::pair<
                                    cpprob::Prior<RealType, cpprob::MetaNormal<double, 0, 10>>,
                                    RealType
                                >
                            >, cpprob::MetaPoisson<int, 10>
                        >
                    >) {
    auto poly = generate_polynomial<RealType>(D);

    for (const auto &point : points) {
        boost::random::normal_distribution<RealType> likelihood{eval_poly(poly, point.first), 1};
        cpprob::observe(likelihood, point.second);
    }
    for (const auto coef : poly) {
        cpprob::predict(coef, "Coefficient");
    }
}

template<class RealType = double, std::size_t N>
void linear_regression(const std::array<std::pair<RealType, RealType>, N> &points,
                    cpprob::Builder<
                        std::array<
                            std::pair<
                                cpprob::Prior<RealType, cpprob::MetaNormal<double, 0, 10>>,
                                RealType
                            >, N
                        >
                    >) {
    using boost::random::normal_distribution;
    normal_distribution<RealType> prior{0,10};
    auto a = cpprob::sample(prior);
    auto b = cpprob::sample(prior);

    for (const auto &point : points) {
        normal_distribution<RealType> likelihood{a*point.first + b, 1};
        cpprob::observe(likelihood, point.second);
    }
    cpprob::predict(a, "a");
    cpprob::predict(b, "b");
}

//-m infer -n 100 -o [[1 2.1] [2 3.9] [3 5.3] [4 7.7] [5 10.2] [6 12.9]]
template<std::size_t D, std::size_t N, class RealType = double>
void poly_adjustment(const std::array<std::array<RealType, 2>, N> &points) {
    auto poly = generate_polynomial<RealType>(D);

    for (const auto &point : points) {
        boost::random::normal_distribution<RealType> likelihood{eval_poly(poly, point[0]), 1};
        cpprob::observe(likelihood, point[1]);
    }
    for (const auto coef : poly) {
        cpprob::predict(coef, "Coefficient");
    }
}

} // end namespace models
#endif //CPPROB_POLY_ADJUSTMENT_HPP
