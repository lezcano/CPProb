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
#include <utility>
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
            std::vector<std::string> trace = get_trace();
            std::string addr = std::accumulate(trace.begin(), trace.end(), std::string(""));

            auto it = Core::_id.insert(std::make_pair(addr, static_cast<int>(Core::_id.size())));

            int id = it.first->second;

            _x[id].emplace_back(static_cast<double>(x));
            return x;
        }

        template<class DistrMath>
        void observe(const DistrMath& distr, double x){
            using std::log;
            using boost::multiprecision::log;
            auto log_prob = log(pdf(distr, x));
            _w += log_prob;
        }

        std::vector<std::string> get_trace(){
            constexpr int buf_size = 1000;
            static void *buffer[buf_size];
            char **strings;

            size_t nptrs = backtrace(buffer, buf_size);

            // We will not store the call to get_traces or the call to observe
            constexpr size_t str_discarded = 2;
            std::vector<std::string> ret;

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
                ret.emplace_back(s.substr(first, last-first));
            }
            free(strings);
            return ret;
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
        std::unordered_map<int, std::vector<double>> _x;
        static std::unordered_map<std::string, int> _id;
    };

    template<typename Func>
    Core eval(Func f){
        using std::exp;
        using boost::multiprecision::exp;
        Core c;
        f(c);
        c._w = exp(c._w);
        return c;
    }

    template<class Func>
    std::unordered_map<int, std::vector<double>>
        expectation(
            const Func& f,
            size_t n = 10000,
            const std::function<std::unordered_map<int, std::vector<double>>(std::unordered_map<int, std::vector<double>>)>& q
            = [](std::unordered_map<int, std::vector<double>> x){return x;}){

        Core core;
        double sum_w;
        std::unordered_map<int, std::vector<double>> ret, aux;

        for(size_t i = 0; i < n; ++i){
            core = eval(f);
            sum_w += core._w;
            aux = q(core._x);

            for(auto& elem : aux){
                // Multiply each element sampled x_i of the trace by the weight of the trace
                std::transform(elem.second.begin(),
                               elem.second.end(),
                               elem.second.begin(),
                               [&](double a){ return core._w * a; });
                auto it = ret.insert(std::make_pair(elem.first, elem.second));
                // If it wasn't inserted then we already had it in the unordered_map
                // Add the vectors
                if(!it.second){
                    bool swap = elem.second.size() > it.first->second.size();
                    if(swap) std::swap(elem.second, it.first->second);
                    std::transform(it.first->second.begin(),
                                   it.first->second.end(),
                                   elem.second.begin(),
                                   it.first->second.begin(),
                                   std::plus<double>());
                }
            }
        }
        // Normalise (Compute E_\pi)
        for(auto& elem : ret)
            std::transform(elem.second.begin(),
                           elem.second.end(),
                           elem.second.begin(),
                           [sum_w](double e){ return e/sum_w; });
        return ret;
    }
}
#endif /* CPPROB_HPP */
