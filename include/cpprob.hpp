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
#include <iostream>

#include <zmq.hpp>
#include <msgpack.hpp>

#include "trace.hpp"

namespace cpprob {

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, double x);

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr);

void set_socket(zmq::socket_t*);
void reset_trace();
Trace get_trace();
void set_training(const bool);
void reset_ids();

void send_observe_init(std::vector<double>&& data);


template<class Func>
void compile(const Func& f, const std::string& tcp_addr = "tcp://*:5556") {
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind (tcp_addr);

    // Setup variables
    set_training(true);
    reset_ids();

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
            reset_trace();
            f();
            auto t = get_trace();
            t.pack(pk);
        }

        zmq::message_t reply (sbuf.size());
        memcpy (reply.data(), sbuf.data(), sbuf.size());
        socket.send (reply);
    }
}


template<class Func>
std::vector<std::vector<double>> inference(const Func& f, std::vector<double> data, size_t n = 100000){
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REQ);
    socket.connect ("tcp://localhost:6668");

    // Setup variables
    set_socket(&socket);
    set_training(false);
    reset_ids();

    send_observe_init(std::move(data));

    double sum_w = 0;
    Trace ret;
    for (size_t i = 0; i < n; ++i) {
        reset_trace();
        f();
        auto t = get_trace();
        auto w = std::exp(t.log_w());
        sum_w += w;
        ret += w*t;
    }
    ret /= sum_w;
    return ret.x();
}
}  // namespace cpprob
#endif  // INCLUDE_CPPROB_HPP_
