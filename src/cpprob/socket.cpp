#include "cpprob/socket.hpp"

#include <iostream>
#include <string>


#include "cpprob/trace.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {


////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////
zmq::context_t context (1);

flatbuffers::FlatBufferBuilder Compilation::buff;
zmq::socket_t Compilation::server (context, ZMQ_REP);
std::vector<flatbuffers::Offset<infcomp::Trace>> Compilation::vec;

void Compilation::connect_server(const std::string& tcp_addr){
    Compilation::server.bind (tcp_addr);
}

int Compilation::get_batch_size(){
    zmq::message_t request;
    Compilation::server.recv (&request);

    auto message = infcomp::GetMessage(request.data());
    auto request_msg = static_cast<const infcomp::TracesFromPriorRequest*>(message->body());

    Compilation::vec.reserve(request_msg->num_traces());
    return request_msg->num_traces();
}

void Compilation::add_trace(const Trace& t){
    Compilation::vec.emplace_back(t.pack(Compilation::buff));;
}

void Compilation::send_batch(){
    auto traces = infcomp::CreateTracesFromPriorReplyDirect(buff, &vec);
    auto msg = infcomp::CreateMessage(
            Compilation::buff,
            infcomp::MessageBody::TracesFromPriorReply,
            traces.Union());
    Compilation::buff.Finish(msg);

    zmq::message_t reply (Compilation::buff.GetSize());
    memcpy (reply.data(), Compilation::buff.GetBufferPointer(), Compilation::buff.GetSize());
    Compilation::server.send (reply);

    Compilation::buff.Clear();
    Compilation::vec.clear();
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

    auto msg = infcomp::CreateMessage(
            buff,
            infcomp::MessageBody::ObservesInitRequest,
            observe_init.Union());

    buff.Finish(msg);
    zmq::message_t request (buff.GetSize());
    memcpy (request.data(), buff.GetBufferPointer(), buff.GetSize());
    Inference::client.send (request);
    buff.Clear();

    // TODO (Lezcano) Is this answer is unnecessary?
    static flatbuffers::FlatBufferBuilder builder_reply;
    zmq::message_t reply;
    Inference::client.recv (&reply);

    auto message = infcomp::GetMessage(reply.data());
    auto reply_msg = static_cast<const infcomp::ObservesInitReply*>(message->body());
    if(!reply_msg->success())
        std::cerr << "Invalid command" << std::endl;
}
}  // namespace cpprob
