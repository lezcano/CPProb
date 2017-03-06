#ifndef INCLUDE_CPPROB_HPP_
#define INCLUDE_CPPROB_HPP_

#include <execinfo.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <cmath>
#include <array>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <utility>

#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <zmq.hpp>
#include <msgpack.hpp>

#include "traits.hpp"
//#include "socket.hpp"

namespace cpprob {
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[ ";
    for (const auto& elem : v)
        out << elem << " ";
    out << "]";
    return out;
}


// Forward declaration to declare the function as a friend of Core
template<class Func>
std::vector<std::vector<double>>
expectation(const Func& f,
            size_t n = 100000,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q
                = [](std::vector<std::vector<double>> x) {return x;});

class Sample{
public:
    Sample(int time_index,
           int sample_instance,
           double value,
           const std::string& proposal_name,
           const std::string& sample_address) :
        time_index_{time_index}, sample_instance_{sample_instance},value_{value},
        proposal_name_{proposal_name}, sample_address_{sample_address}{};

    void pack(msgpack::packer<msgpack::sbuffer>& pk) const {
        pk.pack_map(5);
        pk.pack(std::string("time-index"));
        pk.pack(time_index_);
        pk.pack(std::string("proposal-name"));
        pk.pack(proposal_name_);
        pk.pack(std::string("value"));
        pk.pack(value_);
        pk.pack(std::string("sample-instance"));
        pk.pack(sample_instance_);
        pk.pack(std::string("sample-address"));
        pk.pack(sample_address_);
    }
private:
    // TODO(Lezcano) Have a static counter in this class instead of in Core
    int time_index_;
    int sample_instance_;
    double value_;
    std::string proposal_name_;
    std::string sample_address_;
};

class Core{
public:

    Core(bool training) : training_(training) {}

    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample(const Distr<Params ...>& distr) {
        return sample_impl(distr, false);
    }

    template<template <class ...> class Distr, class ...Params>
    void observe(const Distr<Params ...>& distr, double x) {
        if (training_){
            sample_impl(distr, true);
        }
        else{
            using std::log;
            auto prob = pdf(math_distr(distr), x);
            y_.emplace_back(prob);
            w_ += log(prob);
        }
    }

    std::vector<std::pair<double, int>> x_addr() const {
        return x_addr_;
    }

    std::vector<double> y() const {
        return y_;
    }

    void pack(msgpack::packer<msgpack::sbuffer>& pk){
        pk.pack_map(2);
        pk.pack(std::string("samples"));
        pk.pack_array(samples_.size());
        for(const auto& s : samples_)
            s.pack(pk);
        pk.pack(std::string("observes"));
        pk.pack_map(2);
        pk.pack(std::string("shape"));
        pk.pack_array(1);
        pk.pack(observes_.size());
        pk.pack(std::string("data"));
        pk.pack(observes_);
    }


private:


    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample_impl(const Distr<Params ...>& distr, const bool from_observe) {

        std::string addr = get_addr();

        // TODO(Lezcano) Not parallelizable right now, ids_ is static
        auto id = Core::ids_.emplace(addr, static_cast<int>(Core::ids_.size())).first->second;

        if (id >= static_cast<int>(x_.size()))
            x_.resize(id + 1);

        typename Distr<Params ...>::result_type x;
        if (training_) {
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<boost::random::mt19937, Distr<Params ...>> next_val{rng, distr};
            x = next_val();

            if(from_observe){
                observes_.emplace_back(x);
            }
            else{
                samples_.emplace_back(Sample{time_index_, static_cast<int>(x_[id].size()) + 1, x, "normal", addr});
            }
        } else {
            // Request client
            // TODO(Lezcano) Use proposal distribution to sample given the parameters from the NN
            static boost::random::mt19937 rng{seeded_rng()};
            static boost::random::variate_generator<boost::random::mt19937, Distr<Params ...>> next_val{rng, distr};
            // TODO(Lezcano) Use last_x_ and id to compute x
            // TODO(Lezcano) use std::exchange
            x = next_val();
            last_x_ = x;

            // TODO(Lezcano) Accumulate log(p/q) where q is the proposal distribution
        }

        x_[id].emplace_back(static_cast<double>(x));

        x_addr_.emplace_back(static_cast<double>(x), id);
        ++time_index_;

        return x;
    }

    std::string get_addr() const {
        constexpr int buf_size = 1000;
        static void *buffer[buf_size];
        char **strings;

        size_t nptrs = backtrace(buffer, buf_size);

        // We will not store the call to get_traces or the call to sample
        // We discard either observe -> sample_impl -> get_addr
        //            or     sample  -> sample_impl -> get_addr
        // TODO(Lezcano) check that the compiler does not optimize sample away
        constexpr size_t str_discarded = 3;
        std::vector<std::string> trace;

        strings = backtrace_symbols(buffer, nptrs);
        if (strings == nullptr) {
            perror("backtrace_symbols");
            exit(EXIT_FAILURE);
        }

        // The -2 is to discard the call to _start and
        // the call to __libc_start_main
        for (size_t j = str_discarded; j < nptrs - 2; j++) {
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
    std::enable_if_t<N != 0, T> seeded_rng() const {
        std::array<typename T::result_type, N> random_data;
        std::random_device rd;
        std::generate(random_data.begin(), random_data.end(), std::ref(rd));
        std::seed_seq seeds(random_data.begin(), random_data.end());
        return T{seeds};
    }

    template<class Func>
    friend std::vector<std::vector<double>>
    expectation(const Func&,
                size_t,
                const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>&);

    template<typename Func>
    friend Core eval(Func f, bool training);

    int time_index_ = 1;
    double w_ = 0;
    std::vector<std::vector<double>> x_;
    static std::unordered_map<std::string, int> ids_;

    std::vector<std::pair<double, int>> x_addr_;
    std::vector<double> y_;

    std::vector<Sample> samples_;
    std::vector<double> observes_;

    double last_x_ = 0;

    bool training_;
};

template<typename Func>
Core eval(Func f, bool training) {
    Core c{training};
    f(c);
    c.w_ = std::exp(c.w_);
    return c;
}

template<class Func>
void train(const Func& f, const std::string& tcp_addr = "tcp://*:5556") {
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind (tcp_addr);

    zmq::message_t request;

    //  Wait for next request from client
    //  TODO(Lezcano) look at this code and clean
    while(true){
        socket.recv (&request);
        auto rpl = std::string(static_cast<char*>(request.data()), request.size());
        msgpack::object obj = msgpack::unpack(rpl.data(), rpl.size()).get();
        auto pkg = obj.as<std::map<std::string, msgpack::object>>();
        int batch_size = 0;
        if(pkg["command"].as<std::string>() == "new-batch"){
            batch_size = pkg["command-param"].as<int>();
        }
        else{
            std::cout << "Invalid command " << pkg["command"].as<std::string>() << std::endl;
        }

        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);
        // Array of batch_size sample and observe list
        pk.pack_array(batch_size);

        for (int i = 0; i < batch_size; ++i){
            Core c{true};
            f(c);
            c.pack(pk);
        }

        zmq::message_t reply (sbuf.size());
        memcpy (reply.data(), sbuf.data(), sbuf.size());
        socket.send (reply);
    }
}


// Default parameters declared in forward declaration
template<class Func>
std::vector<std::vector<double>>
expectation(const Func& f,
            size_t n,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q) {
    Core core{false};
    double sum_w = 0;
    std::vector<std::vector<double>> ret, aux;

    Core::ids_.clear();

    for (size_t i = 0; i < n; ++i) {
        core = eval(f, false);
        sum_w += core.w_;
        aux = q(core.x_);

        std::cout << aux << '\t' << core.w_ << std::endl;
        if (aux.size() > ret.size())
            ret.resize(aux.size());

        // Add new trace weighted with its weight
        for (size_t i = 0; i < aux.size(); ++i) {
            if (aux[i].empty()) continue;
            // Multiply each element sampled x_i of the trace by the weight of the trace
            std::transform(aux[i].begin(),
                           aux[i].end(),
                           aux[i].begin(),
                           [&](double a){ return core.w_ * a; });
            // Put in ret[i] the biggest of the two vectors
            if (aux[i].size() > ret[i].size())
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
    for (auto& elem : ret)
        std::transform(elem.begin(),
                       elem.end(),
                       elem.begin(),
                       [sum_w](double e){ return e/sum_w; });

    return ret;
}
}  // namespace cpprob
#endif  // INCLUDE_CPPROB_HPP_
