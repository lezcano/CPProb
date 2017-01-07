#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <random>
#include <iostream>
#include <array>

#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace cpprob{

    template<class RealType = double>
    class Core{
    public:

        template<template <class> class Distr>
        RealType sample(const Distr<RealType>& distr){
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<
                    boost::random::mt19937,
                    Distr<RealType>> next_val{rng, distr};
            auto x = next_val();

            _x.emplace_back(x);
            return x;
        }

        template<template <class, class> class Distr, class Policy>
        void observe(const Distr<RealType, Policy>& distr, RealType x){
            using std::log;
            using boost::multiprecision::log;
            auto log_prob = log(boost::math::pdf(distr, x));
            _w += log_prob;
        }

    private:
        template<class, class> friend class Eval;

        // Idea from
        // http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
        template<class T = boost::random::mt19937, std::size_t N = T::state_size>
        std::enable_if_t<N != 0, T> seeded_rng(){
            std::array<typename T::result_type, N> random_data;
            std::random_device rd;
            std::generate(random_data.begin(), random_data.end(), std::ref(rd));
            std::seed_seq seeds(random_data.begin(), random_data.end());
            return T{seeds};
        }

        std::vector<RealType> _x;
        RealType _w;

    };

    template<class T, class RealType = double>
    class Eval{
    public:

        Eval(T f) : _f{f}{ }

        void operator()() {
            using std::exp;
            using boost::multiprecision::exp;
            _c = Core<RealType>{};
            _f(_c);
            _c._w = exp(_c._w);
        }

        std::vector<RealType> expectation(
                const std::function<std::vector<RealType>(const std::vector<RealType>&)>& q,
                size_t n = 10000){

            // Assume that the vector is constant in size
            // What to do if M^k is different?!
            RealType sum_w = 0;

            this->operator()();
            sum_w += _c._w;
            std::vector<RealType> ret = q(_c._x);
            std::transform(ret.begin(), ret.end(), ret.begin(), [this](RealType e){ return e*_c._w; });

            for(size_t i = 1; i < n; ++i){
                this->operator()();
                sum_w += _c._w;
                auto aux = q(_c._x);
                std::transform(ret.begin(), ret.end(), aux.begin(), ret.begin(),
                                [this](RealType a, RealType b){ return a + _c._w * b; });
            }
            // Normalise (Compute E_\pi)
            std::transform(ret.begin(), ret.end(), ret.begin(),
                            [sum_w](RealType x){ return x/sum_w; });
            return ret;
        }

    private:
        T _f;
        Core<RealType> _c;
    };
}
