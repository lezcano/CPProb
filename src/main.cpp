#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <unordered_map>

#include <boost/random/beta_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/binomial_distribution.hpp>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/bernoulli.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "cpprob.hpp"

using cpprob::Core;

void f(Core& c){
    static const boost::random::beta_distribution<double> beta{1, 1};
    auto x = c.sample(beta);
    //static const boost::random::binomial_distribution<int, RealType> binom{};
    //Core<RealType>::sample(binom);

    const boost::math::bernoulli_distribution<double> ber{x};
    c.observe(ber, 1.0);
}

void g(Core& c){
    constexpr auto n = 6;
    static const boost::random::normal_distribution<double> normal{0, 10};
    static const std::array<std::pair<double, double>, n> arr
                    = {{{1.0, 2.1},
                        {2.0, 3.9},
                        {3.0, 5.3},
                        {4.0, 7.7},
                        {5.0, 10.2},
                        {6.0, 12.9}}};

    const auto slope = c.sample(normal);
    const auto bias = c.sample(normal);
    for(size_t i = 0; i < n; ++i){
        c.observe(
            boost::math::normal_distribution<double>{slope*arr[i].first + bias, 1},
            arr[i].second);
    }
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    for (auto elem : v)
        out << elem << " ";
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

int main(){

    //using RealType = boost::multiprecision::cpp_dec_float_100;
    //using RealType = double;

    std::cout << std::setprecision(10);
    std::cout << "Expectation example 1:\n" << cpprob::expectation(&f) << std::endl;
    std::cout << "Expectation example 2:\n" << cpprob::expectation(&g) << std::endl;
}
