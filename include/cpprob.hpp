#ifndef __CPPROB_HPP
#define __CPPROB_HPP

#include <iostream>
#include <algorithm>
#include <functional>
#include <cmath>
#include <random>
#include <array>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <execinfo.h>

#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace cpprob{

    class Core{
    public:

        template<class DistrRand>
        typename DistrRand::result_type sample(const DistrRand& distr){
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<boost::random::mt19937, DistrRand> next_val{rng, distr};
            auto x = next_val();

            std::string addr = get_addr();

            auto id = Core::_id.emplace(addr, static_cast<int>(Core::_id.size())).first->second;

            if(id >= _x.size())
                _x.resize(id + 1);

            _x[id].emplace_back(static_cast<double>(x));

            return x;
        }

        template<class DistrMath>
        void observe(const DistrMath& distr, double x){
            using std::log;
            auto log_prob = log(pdf(distr, x));
            _w += log_prob;
        }

        std::string get_addr(){
            constexpr int buf_size = 1000;
            static void *buffer[buf_size];
            char **strings;

            size_t nptrs = backtrace(buffer, buf_size);

            // We will not store the call to get_traces or the call to observe
            constexpr size_t str_discarded = 2;
            std::vector<std::string> trace;

            strings = backtrace_symbols(buffer, nptrs);
            if (strings == nullptr) {
                perror("backtrace_symbols");
                exit(EXIT_FAILURE);
            }

            // The -2 is to discard the call to _start and
            // the call to __libc_start_main
            for (int j = str_discarded; j < nptrs - 2; j++){
                std::string s {strings[j]};

                // The +3 is to discard the characters
                auto first = s.find("[0x") + 3;
                auto last = s.find("]");
                trace.emplace_back(s.substr(first, last-first));
            }
            free(strings);
            return std::accumulate(trace.begin(), trace.end(), std::string(""));
        }

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

        double _w = 0;
        std::vector<std::vector<double>> _x;
        static std::unordered_map<std::string, int> _id;
    };

    template<typename Func>
    Core eval(Func f){
        using std::exp;
        Core c;
        f(c);
        c._w = exp(c._w);
        return c;
    }

    template<class Func>
    std::vector<std::vector<double>>
        expectation(
            const Func& f,
            size_t n = 100000,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q
            = [](std::vector<std::vector<double>> x){return x;}){

        Core core;
        double sum_w = 0;
        std::vector<std::vector<double>> ret, aux;

        Core::_id.clear();

        for(size_t i = 0; i < n; ++i){
            core = eval(f);
            sum_w += core._w;
            aux = q(core._x);

            if(aux.size() > ret.size())
                ret.resize(aux.size());

            for(size_t i = 0; i < aux.size(); ++i){
                if(aux[i].empty()) continue;
                // Multiply each element sampled x_i of the trace by the weight of the trace
                std::transform(aux[i].begin(),
                               aux[i].end(),
                               aux[i].begin(),
                               [&](double a){ return core._w * a; });
                // Put in ret[i] the biggest of the two vectors
                if(aux[i].size() > ret[i].size())
                    std::swap(aux[i], ret[i]);

                // Add the vectors
                std::transform(aux[i].begin(),
                               aux[i].end(),
                               ret[i].begin(),
                               ret[i].begin(),
                               std::plus<double>());
            }
        }

        // Normalise (Compute E_\pi)
        for(auto& elem : ret)
            std::transform(elem.begin(),
                           elem.end(),
                           elem.begin(),
                           [sum_w](double e){ return e/sum_w; });

        return ret;
    }
}
#endif /* CPPROB_HPP */
