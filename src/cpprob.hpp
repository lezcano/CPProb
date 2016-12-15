#include <iostream>
#include <random>
#include <algorithm>
#include <functional>
#include <cmath>

#include <boost/math/distributions.hpp>

namespace cpprob{

    class Core{
    public:

        // Currently it does not work for discrete distributions
        template<class Distr>
        double sample(const Distr& distr){
            std::random_device rd;
            std::mt19937 rng{rd()};
            static std::uniform_real_distribution<double> unif{0,1};
            double rand_num = unif(rng);
            double xi = boost::math::quantile(distr, rand_num);

            _x.emplace_back(xi);
            return xi;
        }

        template<class Distr>
        void observe(const Distr& distr, double x){
            auto log_prob = std::log(boost::math::pdf(distr, x));
            _w += log_prob;
        }

    private:
        template<class T> friend class Eval;

        std::vector<double> _x;
        double _w;

    };

    template<class T>
    class Eval{
    public:

        Eval(T f) : _f{f}{ }

        void operator()(bool print=true) {
            _c._w = 0;
            _c._x.resize(0);
            _f(_c);
            _c._w = std::exp(_c._w);

            if(print){
                std::cout << "x:";
                for(auto x : _c._x)
                    std::cout << " " << x;
                std::cout << '\n';

                std::cout << "w: " << _c._w << '\n';
            }
        }

        std::vector<double> expectation(
                const std::function<std::vector<double>(const std::vector<double>&)>& q,
                size_t n = 10000){


            // Assume that the vector is constant in size
            // What to do if M^k is different?!
            //
            double sum_w = 0;
            std::vector<double> aux;

            this->operator()(false);
            sum_w += _c._w;
            std::vector<double> exp = q(_c._x);
            std::transform(exp.begin(), exp.end(), exp.begin(), [this](double e){ return e*_c._w; });

            for(size_t i = 1; i < n; ++i){
                this->operator()(false);
                sum_w += _c._w;
                aux = q(_c._x);
                std::transform(exp.begin(), exp.end(), aux.begin(), exp.begin(),
                                [this](double a, double b){ return a + _c._w * b; });
            }
            // Normalise (Compute E_\pi instead of E_\gamma)
            std::transform(exp.begin(), exp.end(), exp.begin(),
                            [sum_w](double x){ return x/sum_w; });
            return exp;
        }

    private:
        T _f;
        Core _c;
    };
}
