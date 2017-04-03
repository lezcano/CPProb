#ifndef INCLUDE_CPPROB_HPP_
#define INCLUDE_CPPROB_HPP_

#include <execinfo.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <cmath>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <utility>

#include <zmq.hpp>
#include <msgpack.hpp>

namespace cpprob {

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[ ";
    for (const auto& elem : v)
        out << elem << " ";
    out << "]";
    return out;
}

class Core;

template<typename Func>
Core eval(Func f, bool training, zmq::socket_t* socket=nullptr);

template<class Func>
std::vector<std::vector<double>>
expectation(const Func& f,
            std::vector<double> data,
            size_t n = 100000,
            const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q
                = [](std::vector<std::vector<double>> x) {return x;});

class Sample{
public:
    Sample(int time_index,
           int sample_instance,
           double value,
           const std::string& proposal_name,
           const std::string& sample_address);

    void pack(msgpack::packer<msgpack::sbuffer>& pk) const;

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

    Core(bool training, zmq::socket_t* socket = nullptr);

    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample(const Distr<Params ...>& distr);

    template<template <class ...> class Distr, class ...Params>
    void observe(const Distr<Params ...>& distr, double x);

    void pack(msgpack::packer<msgpack::sbuffer>& pk);


private:

    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample_distr(const Distr<Params ...>& distr);

    template<template <class ...> class Distr, class ...Params>
    typename Distr<Params ...>::result_type sample_impl(const Distr<Params ...>& distr, const bool from_observe);

    template<class Func>
    friend std::vector<std::vector<double>>
    expectation(const Func&,
                std::vector<double>,
                size_t,
                const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>&);

    template<typename Func>
    friend Core eval(Func f, bool training, zmq::socket_t* socket);

    int time_index_ = 1;
    double w_ = 0;
    std::vector<std::vector<double>> x_;
    static std::unordered_map<std::string, int> ids_;

    std::vector<std::pair<double, int>> x_addr_;
    std::vector<double> y_;

    std::vector<Sample> samples_;
    std::vector<double> observes_;

    bool training_;
    zmq::socket_t* socket_ = nullptr;

    int prev_sample_instance_ = 0;
    double prev_x_ = 0;
    std::string prev_addr_ = "";
};



template<typename Func>
Core eval(Func f, bool training, zmq::socket_t* socket) {
    Core c = Core{training, socket};
    f(c);
    c.w_ = std::exp(c.w_);
    return c;
}

// Default parameters declared in forward declaration
//
template<class Func>
std::vector<std::vector<double>> expectation(
        const Func& f,
        std::vector<double> data,
        size_t n,
        const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>& q){
    Core core{false};
    double sum_w = 0;
    std::vector<std::vector<double>> ret, aux;

    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);
    socket.connect ("tcp://localhost:6668");

    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(&sbuf);

    pk.pack_map(2);
        pk.pack(std::string("command"));
        pk.pack(std::string("observe-init"));

        pk.pack(std::string("command-param"));
        pk.pack_map(2);
            pk.pack(std::string("shape"));
            pk.pack_array(1);
                pk.pack(data.size());

            pk.pack(std::string("data"));
            pk.pack(data);

    zmq::message_t request (sbuf.size()), reply;
    memcpy (request.data(), sbuf.data(), sbuf.size());
    socket.send (request);

    // TODO (Lezcano) This answer is unnecessary
    socket.recv (&reply);
    auto rpl = std::string(static_cast<char*>(reply.data()), reply.size());

    std::string answer = msgpack::unpack(rpl.data(), rpl.size()).get().as<std::string>();
    if(answer != "observe-received")
        std::cout << "Invalid command " << answer << std::endl;

    Core::ids_.clear();

    for (size_t i = 0; i < n; ++i) {
        core = eval(f, false, &socket);
        sum_w += core.w_;
        aux = q(core.x_);

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
