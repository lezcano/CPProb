#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>

#include <boost/random/beta_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/bernoulli.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "cpprob.hpp"

using RealType = boost::multiprecision::cpp_dec_float_100;
//using RealType = double;

void f(cpprob::Core<RealType>& c){
    static const boost::random::beta_distribution<RealType> beta{1, 1};
    auto x = c.sample(beta);

    const boost::math::bernoulli_distribution<RealType> ber{x};
    c.observe(ber, 1.0);
}

void g(cpprob::Core<RealType>& c){
    constexpr auto n = 6;
    static const boost::random::normal_distribution<RealType> normal{0, 10};
    static const std::array<std::pair<RealType, RealType>, n> arr
                    = {{{1.0, 2.1},
                        {2.0, 3.9},
                        {3.0, 5.3},
                        {4.0, 7.7},
                        {5.0, 10.2},
                        {6.0, 12.9}}};

    const auto slope = c.sample(normal);
    const auto bias = c.sample(normal);
    for(size_t i = 0; i < n; ++i){
        c.observe(boost::math::normal_distribution<RealType>{slope*arr[i].first + bias, 1}, arr[i].second);
    }
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    for (auto elem : v)
        out << elem << " ";
    return out;
}

int main(){
    cpprob::Eval<decltype(&f), RealType> e1{&f};
    cpprob::Eval<decltype(&g), RealType> e2{&g};

    std::cout << std::setprecision(10);
    std::cout << "Expectation example 1: " << e1.expectation([](auto x){return x;}) << std::endl;
    std::cout << "Expectation example 2: " << e2.expectation([](auto x){return x;}) << std::endl;
}
