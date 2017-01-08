#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <random>
#include <array>
#include <type_traits>

#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace cpprob{

    template<class RealType = double>
    class Core{
    public:

        template<class DistrRand>
        typename DistrRand::result_type sample(const DistrRand& distr){
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<boost::random::mt19937, DistrRand> next_val{rng, distr};
            auto x = next_val();

            _x.emplace_back(static_cast<RealType>(x));
            return x;
        }

        template<class DistrMath>
        void observe(const DistrMath& distr, RealType x){
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
        RealType _w = 0;

    };

    template<class T, class RealType = double>
    class Eval{
    public:

        Eval(T f) : _f{f}{ }

        std::vector<RealType> expectation(
                const std::function<std::vector<RealType>(std::vector<RealType>)>& q
                    = [](std::vector<RealType> x) -> std::vector<RealType>{return x;},
                size_t n = 10000){

            // Assume that the vector is constant in size
            // What to do if M^k is different?!
            // Maybe Macros & __COUNTER__ or BOOST_PP_COUNTER
            RealType sum_w = 0;

            auto core = this->eval_f();
            sum_w += core._w;
            std::vector<RealType> ret = q(core._x);
            std::vector<RealType> aux;
            std::transform(ret.begin(), ret.end(), ret.begin(), [&](RealType e){ return e*core._w; });

            for(size_t i = 1; i < n; ++i){
                core = this->eval_f();
                sum_w += core._w;
                aux = q(core._x);
                std::transform(ret.begin(), ret.end(), aux.begin(), ret.begin(),
                                [&](RealType a, RealType b){ return a + core._w * b; });
            }
            // Normalise (Compute E_\pi)
            std::transform(ret.begin(), ret.end(), ret.begin(),
                            [sum_w](RealType x){ return x/sum_w; });
            return ret;
        }

    private:

        Core<RealType>  eval_f(){
            using std::exp;
            using boost::multiprecision::exp;
            Core<RealType> core;
            _f(core);
            core._w = exp(core._w);
            return core;
        }

        T _f;
    };
}
