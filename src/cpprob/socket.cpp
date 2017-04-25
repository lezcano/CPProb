#include "cpprob/socket.hpp"

#include <iostream>
#include <string>


#include "cpprob/trace.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

flatbuffers::FlatBufferBuilder Compilation::buff;

zmq::context_t context (1);
zmq::socket_t server (context, ZMQ_REP);
std::vector<flatbuffers::Offset<infcomp::Trace>> vec;

void Compilation::connect_server(const std::string& tcp_addr){
    server.bind (tcp_addr);
}

int Compilation::get_batch_size(){
    zmq::message_t request;
    server.recv (&request);

    auto reply = flatbuffers::GetRoot<infcomp::TracesFromPriorRequest>(request.data());
    vec.reserve(reply->num_traces());
    return reply->num_traces();
}

void Compilation::add_trace(const Trace& t){
    vec.emplace_back(t.pack(Compilation::buff));;
}

void Compilation::send_batch(){
    auto traces = infcomp::CreateTracesFromPriorReplyDirect(buff, &vec);
    infcomp::CreateMessage(
            Compilation::buff,
            infcomp::MessageBody::TracesFromPriorReply,
            traces.Union());
    zmq::message_t reply (Compilation::buff.GetSize());
    memcpy (reply.data(), Compilation::buff.GetBufferPointer(), Compilation::buff.GetSize());
    server.send (reply);
    Compilation::buff.Clear();
    vec.clear();
}
////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

zmq::socket_t Inference::client (context, ZMQ_REQ);

void Inference::connect_client(const std::string& tcp_addr){
    Inference::client.connect (tcp_addr);
}

void Inference::send_observe_init(std::vector<double>&& data){
    static flatbuffers::FlatBufferBuilder buff;

    const auto shape = std::vector<int>{static_cast<int>(data.size())};
    auto observe_init = infcomp::CreateObservesInitRequest(
            buff,
            infcomp::CreateNDArrayDirect(buff, &data, &shape));

    infcomp::CreateMessage(
            buff,
            infcomp::MessageBody::ObservesInitRequest,
            observe_init.Union());

    zmq::message_t request (buff.GetSize());
    memcpy (request.data(), buff.GetBufferPointer(), buff.GetSize());
    Inference::client.send (request);
    buff.Clear();

    // TODO (Lezcano) Is this answer is unnecessary?
    static flatbuffers::FlatBufferBuilder builder_reply;
    zmq::message_t reply;
    Inference::client.recv (&reply);

    auto reply_msg = flatbuffers::GetRoot<infcomp::ObservesInitReply>(reply.data());
    if(!reply_msg->success())
        std::cerr << "Invalid command " << std::endl;
}
}  // namespace cpprob
