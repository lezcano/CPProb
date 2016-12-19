#include <iostream>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/bernoulli.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include "cpprob.hpp"

using RealType = boost::multiprecision::cpp_dec_float_100;

void f(cpprob::Core<RealType>& c){
    static boost::math::beta_distribution<RealType> beta{1, 1};
    auto x = c.sample(beta);

    boost::math::bernoulli_distribution<RealType> ber{x};
    c.observe(ber, 1.0);
}

void g(cpprob::Core<RealType>& c){
    constexpr auto n = 6;
    static boost::math::normal_distribution<RealType> normal{0, 10};
    static std::array<RealType, 2*n> arr
        = { 1.0, 2.1, 2.0, 3.9,  3.0, 5.3,
            4.0, 7.7, 5.0, 10.2, 6.0, 12.9};

    auto slope = c.sample(normal);
    auto bias = c.sample(normal);
    for(size_t i = 0; i < n; ++i){
        c.observe(boost::math::normal_distribution<RealType>{slope*arr[2*i] + bias, 1}, arr[2*i+1]);
    }

}

int main(){
    cpprob::Eval<decltype(&f), RealType> e{&f};
    cpprob::Eval<decltype(&g), RealType> e2{&g};
    /*
    for(size_t i = 0; i < 10; ++i) e();

    for(size_t i = 0; i < 10; ++i) e2();
    */

    auto expc = e.expectation([](auto x){return x;});
    std::cout << "Expectation:";

   std::cout << std::setprecision(std::numeric_limits<RealType>::max_digits10);
    for(auto elem : expc){
        std::cout << " " << elem;
    }
    std::cout << '\n';
}
