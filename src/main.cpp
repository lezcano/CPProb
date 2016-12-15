#include <iostream>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/bernoulli.hpp>

#include "cpprob.hpp"

void f(cpprob::Core& c){
    static boost::math::beta_distribution<double> beta{1, 1};
    auto x = c.sample(beta);
    double acc = 0;

    boost::math::bernoulli ber{x};
    c.observe(ber, 1);
}

void g(cpprob::Core& c){
    constexpr auto n = 6;
    static boost::math::normal normal{0, 10};
    static std::array<double, 2*n> arr
        = { 1.0, 2.1, 2.0, 3.9,  3.0, 5.3,
            4.0, 7.7, 5.0, 10.2, 6.0, 12.9};

    auto slope = c.sample(normal);
    auto bias = c.sample(normal);
    for(size_t i = 0; i < n; ++i){
        c.observe(boost::math::normal{slope*arr[2*i] + bias, 1}, arr[2*i+1]);
    }

}

int main(){
    cpprob::Eval<decltype(&f)> e{&f};
    cpprob::Eval<decltype(&g)> e2{&g};
    /*
    for(size_t i = 0; i < 10; ++i) e();

    for(size_t i = 0; i < 10; ++i) e2();
    */

    auto expc = e.expectation([](auto x){return x;});
    std::cout << "Expectation:";
    for(auto elem : expc){
        std::cout << " " << elem;
    }
    std::cout << '\n';
}
