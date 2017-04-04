#ifndef INCLUDE_CPPROB_HPP_
#define INCLUDE_CPPROB_HPP_

#include <execinfo.h>

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <cmath>

#include "trace.hpp"
#include "socket.hpp"

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

template<class Func>
void compile(const Func& f, const std::string& tcp_addr = "tcp://*:5556") {
    connect_server(tcp_addr);
    set_training(true);

    while(true){
        auto batch_size = get_batch_size();

        for (int i = 0; i < batch_size; ++i){
            reset_trace();
            f();
            add_trace(get_trace(), i);
        }
        send_batch();
    }
}


template<class Func>
std::vector<std::vector<double>> inference(
        const Func& f,
        std::vector<double> data,
        const std::string& tcp_addr = "tcp://localhost:6668",
        size_t n = 100000){

    connect_client(tcp_addr);
    set_training(false);

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
