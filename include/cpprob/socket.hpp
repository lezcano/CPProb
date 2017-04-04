#ifndef INCLUDE_SOCKET_HPP_
#define INCLUDE_SOCKET_HPP_

#include <string>

#include <msgpack.hpp>
#include <zmq.hpp>

#include "trace.hpp"

namespace cpprob{

////////////////////////////////////////////////////////////////////////////////
/////////////////////////            Utils              ////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class T>
T receive(zmq::socket_t& socket){
    zmq::message_t reply;
    socket.recv (&reply);

    std::string str = std::string(static_cast<char*>(reply.data()), reply.size());
    msgpack::object_handle oh = msgpack::unpack(str.data(), str.size());
    return oh.get().as<T>();
}

template<class T>
void send(T&& obj, zmq::socket_t& socket){
    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, obj);

    zmq::message_t request (sbuf.size());
    memcpy (request.data(), sbuf.data(), sbuf.size());
    socket.send (request);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

void connect_server(const std::string& tcp_addr);
int get_batch_size();
void add_trace(const Trace& t, int num);
void send_batch();

////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

void connect_client(const std::string& tcp_addr);
void send_observe_init(std::vector<double>&& data);
std::vector<double> get_params(const SampleInference& curr_sample, const PrevSampleInference& prev_sample);

}       // namespace cpprob
#endif  // INCLUDE_SOCKET_HPP_
