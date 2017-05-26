#ifndef INCLUDE_SOCKET_HPP_
#define INCLUDE_SOCKET_HPP_

#include <exception>
#include <string>
#include <vector>

#include <zmq.hpp>

#include "cpprob/trace.hpp"
#include "cpprob/distribution_utils.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob{

// Forward Declaration
template<class T>
class NDArray;

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class Compilation {
public:
    static void connect_server(const std::string & tcp_addr);
    static std::size_t get_batch_size();
    static void add_trace(const TraceCompile & t);
    static void send_batch();
    static void config_to_file(const std::string & dump_folder);
private:
    // Static data
    static flatbuffers::FlatBufferBuilder buff;
    static std::vector<flatbuffers::Offset<infcomp::protocol::Trace>> vec;

    static bool to_file;
    static std::string dump_folder;

    static zmq::socket_t server;

};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class Inference {
public:
    static void connect_client(const std::string& tcp_addr);
    static void send_observe_init(const NDArray<double> & data);

    template<template <class ...> class Distr, class ...Params>
    static auto get_proposal(const Sample& curr_sample, const Sample& prev_sample){
        static flatbuffers::FlatBufferBuilder buff;

        auto msg = infcomp::protocol::CreateMessage(
                buff,
                infcomp::protocol::MessageBody::ProposalRequest,
                infcomp::protocol::CreateProposalRequest(buff, curr_sample.pack(buff), prev_sample.pack(buff)).Union());

        buff.Finish(msg);

        zmq::message_t request (buff.GetSize());
        memcpy (request.data(), buff.GetBufferPointer(), buff.GetSize());
        client.send(request);
        buff.Clear();

        zmq::message_t reply;
        client.recv (&reply);

        auto message = infcomp::protocol::GetMessage(reply.data());
        auto reply_msg = static_cast<const infcomp::protocol::ProposalReply*>(message->body());
        // TODO(Lezcano) C++17 std::expected would solve this in a cleaner way
        if (!reply_msg->success()) {
            throw std::runtime_error("NN could not propose parameters.");
        }
        return proposal<Distr, Params...>::get_distr(reply_msg);
    }

private:
    static zmq::socket_t client;
};
}       // namespace cpprob
#endif  // INCLUDE_SOCKET_HPP_
