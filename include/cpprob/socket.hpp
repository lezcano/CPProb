#ifndef INCLUDE_SOCKET_HPP_
#define INCLUDE_SOCKET_HPP_

#include <string.h>                                     // for memcpy
#include <cstddef>                                      // for size_t
#include <stdexcept>                                    // for runtime_error
#include <string>                                       // for string
#include <unordered_map>                                // for unordered_map
#include <utility>                                      // for pair
#include <vector>                                       // for vector

#include <zmq.hpp>                                      // for message_t

#include "cpprob/any.hpp"                               // for any
#include "cpprob/distributions/distribution_utils.hpp"  // for proposal
#include "cpprob/ndarray.hpp"                           // for NDArray
#include "cpprob/sample.hpp"                            // for Sample

#include "flatbuffers/infcomp_generated.h"              // for CreateMessage
namespace cpprob { class TraceCompile; }

namespace cpprob{

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class SocketCompile {
public:
    static void connect_server(const std::string & tcp_addr);
    static void config_file(const std::string & dump_folder);

    static std::size_t get_batch_size();

private:
    // Static Attributes
    static zmq::socket_t server_;
    static std::string dump_folder_;

    // Friends
    friend class StateCompile;
    static void send_batch(const std::vector<TraceCompile> & traces);
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class SocketInfer {
public:
    static void connect_client(const std::string& tcp_addr);
    static void config_file(const std::string & dump_file);

    static void send_observe_init(const NDArray<double> & data);

    template<template <class ...> class Distr, class ...Params>
    static auto get_proposal(const Sample& curr_sample, const Sample& prev_sample){
        static flatbuffers::FlatBufferBuilder buff;

        auto msg = infcomp::protocol::CreateMessage(
                buff,
                infcomp::protocol::MessageBody::ProposalRequest,
                infcomp::protocol::CreateProposalRequest(buff, curr_sample.pack(buff), prev_sample.pack(buff)).Union());

        buff.Finish(msg);

        zmq::message_t request {buff.GetSize()};
        memcpy(request.data(), buff.GetBufferPointer(), buff.GetSize());
        client_.send(request);
        buff.Clear();

        zmq::message_t reply;
        client_.recv(&reply);

        auto message = infcomp::protocol::GetMessage(reply.data());
        auto reply_msg = static_cast<const infcomp::protocol::ProposalReply*>(message->body());
        // TODO(Lezcano) C++17 std::expected would solve this in a cleaner way
        if (!reply_msg->success()) {
            throw std::runtime_error("NN could not propose parameters.");
        }
        return proposal<Distr, Params...>::get_distr(reply_msg);
    }

private:
    friend class StateInfer;

    static void dump_predicts(const std::vector<std::pair<int, cpprob::any>> & predicts, const double log_w, const std::string & suffix);

    static void dump_ids(const std::unordered_map<std::string, int> & ids_predict);
    static void delete_file(const std::string & suffix);

    static zmq::socket_t client_;
    static std::string dump_file_;
};
}       // namespace cpprob
#endif  // INCLUDE_SOCKET_HPP_
