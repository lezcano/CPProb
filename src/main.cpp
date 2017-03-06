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

template<class RealType = double>
void f(cpprob::Core& c) {
    using boost::random::beta_distribution;
    using boost::random::bernoulli_distribution;
    using boost::random::binomial_distribution;

    static const beta_distribution<RealType> beta{1, 1};
    auto x = c.sample(beta);

    const bernoulli_distribution<RealType> ber{x};
    c.observe(ber, 1.0);
}

template<class RealType = double>
void g(cpprob::Core& c) {
    using boost::random::normal_distribution;

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

void mean_normal(cpprob::Core& c) {
    using boost::random::normal_distribution;
    static const normal_distribution<> normal{0, 1};
    const double y1 = 0.2, y2 = 0.2;
    auto mean = c.sample(normal);

    c.observe(normal_distribution<>{mean, 1}, y1);
    c.observe(normal_distribution<>{mean, 1}, y2);
}

/*
template<typename T>
std::vector<T>& operator*=(std::vector<T>& v, double a)
{
    std::transform(v.begin(),
                   v.end(),
                   v.begin(),
                   [a](double elem){ return elem * a; });
    return v;
}

template<typename T>
std::vector<double>& operator+=(std::vector<double>& v1, const std::vector<double>& v2)
{
    if(v2.size() > v1.size())
        v1.resize(v2.size());

    std::transform(v2.begin(),
                   v2.end(),
                   v1.begin(),
                   v1.begin(),
                   std::plus<double>());
    return v1;
}
*/

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
    train(&mean_normal, "tcp://*:5556");
    //std::cout << std::setprecision(10);
    //std::cout << "Expectation example means:\n" << cpprob::expectation(&mean_normal) << std::endl;
    //std::cout << "Expectation example 1:\n" << cpprob::expectation(&f<>) << std::endl;
    //std::cout << "Expectation example 2:\n" << cpprob::expectation(&g<>) << std::endl;
}
