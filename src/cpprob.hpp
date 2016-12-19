#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <random>

#include <boost/math/distributions.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace cpprob{

    template<class RealType = double>
    class Core{
    public:

        // Currently it does not work for discrete distributions
        // Look for default template parameters in deduction template functions
        template<template <class, class> class Distr, class Policy>
        RealType sample(const Distr<RealType, Policy>& distr){
            std::random_device rd;
            boost::random::mt19937 rng{rd()};
            static boost::random::uniform_real_distribution<RealType> unif{0,1};
            auto rand_num = unif(rng);
            auto xi = boost::math::quantile(distr, rand_num);

            _x.emplace_back(xi);
            return xi;
        }

        template<template <class, class> class Distr, class Policy>
        void observe(const Distr<RealType, Policy>& distr, RealType x){
            using std::log;
            using boost::multiprecision::log;
            auto log_prob = log(boost::math::pdf(distr, x));
            _w += log_prob;
        }

    private:
        template<class U, class V> friend class Eval;

        std::vector<RealType> _x;
        RealType _w;

    };

    template<class T, class RealType = double>
    class Eval{
    public:

        Eval(T f) : _f{f}{ }

        void operator()(bool print=true) {
            using std::exp;
            using boost::multiprecision::exp;
            _c._w = 0;
            _c._x.resize(0);
            _f(_c);
            _c._w = exp(_c._w);

            if(print){
                std::cout << "x:";
                for(auto x : _c._x)
                    std::cout << " " << x;
                std::cout << '\n';

                std::cout << "w: " << _c._w << '\n';
            }
        }

        std::vector<RealType> expectation(
                const std::function<std::vector<RealType>(const std::vector<RealType>&)>& q,
                size_t n = 10000){


            // Assume that the vector is constant in size
            // What to do if M^k is different?!
            //
            RealType sum_w = 0;
            std::vector<RealType> aux;

            this->operator()(false);
            sum_w += _c._w;
            std::vector<RealType> exp = q(_c._x);
            std::transform(exp.begin(), exp.end(), exp.begin(), [this](RealType e){ return e*_c._w; });

            for(size_t i = 1; i < n; ++i){
                this->operator()(false);
                sum_w += _c._w;
                aux = q(_c._x);
                std::transform(exp.begin(), exp.end(), aux.begin(), exp.begin(),
                                [this](RealType a, RealType b){ return a + _c._w * b; });
            }
            // Normalise (Compute E_\pi instead of E_\gamma)
            std::transform(exp.begin(), exp.end(), exp.begin(),
                            [sum_w](RealType x){ return x/sum_w; });
            return exp;
        }

    private:
        T _f;
        Core<RealType> _c;
    };
}
