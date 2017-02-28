#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>    // std::sqrt
#include <unordered_map>
#include <utility>

#include <boost/random/beta_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "cpprob.hpp"

void f(cpprob::Core<false>& c) {
    using boost::random::beta_distribution;
    using boost::random::bernoulli_distribution;
    using boost::random::binomial_distribution;
    using RealType = double;

    static const beta_distribution<RealType> beta{1, 1};
    auto x = c.sample(beta);

    const bernoulli_distribution<RealType> ber{x};
    c.observe(ber, 1.0);
}

void g(cpprob::Core<false>& c) {
    using boost::random::normal_distribution;
    using RealType = double;

    constexpr auto n = 6;
    static const normal_distribution<RealType> normal{0, 10};
    static const std::array<std::pair<RealType, RealType>, n> arr
                    = {{{1.0, 2.1},
                        {2.0, 3.9},
                        {3.0, 5.3},
                        {4.0, 7.7},
                        {5.0, 10.2},
                        {6.0, 12.9}}};

    const auto slope = c.sample(normal);
    const auto bias = c.sample(normal);
    for (size_t i = 0; i < n; ++i) {
        c.observe(
            normal_distribution<RealType>{slope*arr[i].first + bias, 1},
            arr[i].second);
    }
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[ ";
    for (const auto& elem : v)
        out << elem << " ";
    out << "]";
    return out;
}


template<typename T, typename U>
std::ostream& operator<< (std::ostream& out, const std::unordered_map<T, U>& v) {
    out << "{" << '\n';
    for (const auto& elem : v)
        out << elem.first << ": " << elem.second << '\n';
    out << "}";
    return out;
}

int main() {
    std::cout << std::setprecision(10);
    std::cout << "Expectation example 1:\n" << cpprob::expectation(&f) << std::endl;
    std::cout << "Expectation example 2:\n" << cpprob::expectation(&g) << std::endl;
}
