#include "gtest/gtest.h"

#include <cmath>    // exp, log
#include <numeric>  // isinf
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/uniform.hpp>

#include "cpprob/distributions/utils_distributions.hpp"

using std::log;
using std::log;
using boost::math::pdf;
using cpprob::logpdf;

constexpr double eps = 1e-8;

template<class RealType>
auto cast(const boost::math::normal_distribution<RealType> & d) {
    return boost::random::normal_distribution<RealType>(d.mean(), d.standard_deviation());
}

TEST(logpdf, normal) {
    for (int mean = -10; mean < 10; ++mean) {
        for (int std = 1; std < 20; ++std) {
            for (int f = 1; f < 20; ++f) {
                boost::math::normal_distribution<> distr(mean / static_cast<double>(f), std);
                for (int i = -10; i < 10; ++i) {
                    EXPECT_NEAR(log(pdf(distr, i)), logpdf<decltype(cast(distr))>()(cast(distr), i), eps);
                    EXPECT_NEAR(pdf(distr, i), exp(logpdf<decltype(cast(distr))>()(cast(distr), i)), eps);
                }
            }
        }
    }
}

template<class RealType>
auto cast(const boost::math::poisson_distribution<RealType> & d) {
    return boost::random::poisson_distribution<RealType>(d.mean());
}


TEST(logpdf, poisson) {
    for (int mean = 0; mean < 10; ++mean) {
        for (int f = 1; f < 20; ++f) {
            boost::math::normal_distribution<> distr(mean / static_cast<double>(f));
            for (int i = 0; i < 20; ++i) {
                EXPECT_NEAR(log(pdf(distr, i)), logpdf<decltype(cast(distr))>()(cast(distr), i), eps);
                EXPECT_NEAR(pdf(distr, i), exp(logpdf<decltype(cast(distr))>()(cast(distr), i)), eps);
            }
        }
    }
}

template<class RealType>
auto cast(const boost::math::uniform_distribution<RealType> & d) {
    return boost::random::uniform_real_distribution<RealType>(d.lower(), d.upper());
}


TEST(logpdf, uniform) {
    for (int a = -10; a < 10; ++a) {
        for (int b = a+1; b < 10; ++b) {
            for (int f = 1; f < 20; ++f) {
                boost::math::uniform_distribution<> distr(a/static_cast<double>(f), b/static_cast<double>(f));
                for (int i = -10; i < 10; ++i) {
                    EXPECT_NEAR(pdf(distr, i), exp(logpdf<decltype(cast(distr))>()(cast(distr), i)), eps);
                    // EXPECT_NEAR doesn't handle well the -inf values
                    if (std::isinf(log(pdf(distr, i)))  &&
                        std::isinf(logpdf<decltype(cast(distr))>()(cast(distr), i))) {
                        continue;
                    }
                    EXPECT_NEAR(log(pdf(distr, i)), logpdf<decltype(cast(distr))>()(cast(distr), i), eps);
                }
            }
        }
    }
}

