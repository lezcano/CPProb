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
#include "cpprob/ndarray.hpp"                           // for NDArray
#include "cpprob/sample.hpp"                            // for Sample
#include "flatbuffers/infcomp_generated.h"

namespace cpprob { class TraceCompile; }

namespace cpprob{

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class SocketCompile {
public:
    static void connect_server(const std::string & tcp_addr);
    static void config_file(const std::string & dump_folder);

    static int get_batch_size();
    static void send_finish_compilation();

private:
    // Static Attributes
    static zmq::socket_t server_;
    static std::string dump_folder_;

    // Friends
    friend class StateCompile;

    // Function to send the batch from StateCompile
    static void send_batch(const flatbuffers::FlatBufferBuilder & buff);
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

class SocketInfer {
public:
    static void connect_client(const std::string& tcp_addr);
    static void config_file(const std::string & dump_file);

    static void send_start_inference(const flatbuffers::FlatBufferBuilder & buff);
    static void send_finish_inference();

    template<class Proposal>
    static auto get_proposal(const flatbuffers::FlatBufferBuilder & buff){
        zmq::message_t request {buff.GetSize()};
        memcpy(request.data(), buff.GetBufferPointer(), buff.GetSize());
        client_.send(request);

        zmq::message_t reply;
        client_.recv(&reply);

        auto message = protocol::GetMessage(reply.data());
        auto reply_msg = message->body_as_ReplyProposal();
        // TODO(Lezcano) C++17 std::expected would solve this in a cleaner way
        if (reply_msg->distribution_type() == protocol::Distribution::NONE) {
            throw std::runtime_error("NN could not propose parameters.");
        }

        auto distr = reply_msg->template distribution_as<buffer_t<Proposal>>();
        if (distr == nullptr) {
            throw std::runtime_error("NN proposed parameters for an incorrect proposal distribution.");
        }

        return from_flatbuffers<Proposal>()(distr);
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
