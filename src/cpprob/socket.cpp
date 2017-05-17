#include "cpprob/socket.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <cstdio>       // std::fwrite
#include <exception>    // std::terminate

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>

#include "cpprob/trace.hpp"
#include "flatbuffers/infcomp_generated.h"


namespace cpprob {


////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////
zmq::context_t context (1);

// Static data
flatbuffers::FlatBufferBuilder Compilation::buff;
std::vector<flatbuffers::Offset<infcomp::protocol::Trace>> Compilation::vec;

bool Compilation::to_file;
int Compilation::batch_size;
std::string Compilation::dump_folder;

zmq::socket_t Compilation::server (context, ZMQ_REP);

void Compilation::connect_server(const std::string& tcp_addr)
{
    Compilation::server.bind (tcp_addr);
    to_file = false;
}

void Compilation::config_to_file(const std::string & dump_folder)
{
    Compilation::dump_folder = dump_folder;
    to_file = true;
}

std::size_t Compilation::get_batch_size()
{
    zmq::message_t request;
    Compilation::server.recv(&request);

    auto message = infcomp::protocol::GetMessage(request.data());
    auto request_msg = static_cast<const infcomp::protocol::TracesFromPriorRequest *>(message->body());

    Compilation::vec.reserve(static_cast<size_t>(request_msg->num_traces()));
    return request_msg->num_traces();
}

void Compilation::add_trace(const TraceCompile& t)
{
    Compilation::vec.emplace_back(t.pack(Compilation::buff));;
}

void Compilation::send_batch()
{
    auto traces = infcomp::protocol::CreateTracesFromPriorReplyDirect(buff, &vec);
    auto msg = infcomp::protocol::CreateMessage(
            Compilation::buff,
            infcomp::protocol::MessageBody::TracesFromPriorReply,
            traces.Union());
    Compilation::buff.Finish(msg);

    if (to_file) {
        static auto rand_gen {boost::uuids::random_generator()};

        // We already know that dump_folder exists
        auto file_name = dump_folder + "/" + boost::lexical_cast<std::string>(rand_gen());
        auto file = std::ofstream(file_name, std::ofstream::binary);
        if (!file.is_open()) {
            std::cerr << "File \"" << file_name << "\" could not be open.\n";
            std::terminate();
        }
        file.write(reinterpret_cast<char *>(Compilation::buff.GetBufferPointer()), Compilation::buff.GetSize());
    }
    else {
        zmq::message_t reply(Compilation::buff.GetSize());
        memcpy(reply.data(), Compilation::buff.GetBufferPointer(), Compilation::buff.GetSize());
        Compilation::server.send(reply);
    }

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

void Inference::send_observe_init(const NDArray<double> & data) {
    static flatbuffers::FlatBufferBuilder buff;

    auto observe_init = infcomp::protocol::CreateObservesInitRequest(
            buff,
            infcomp::protocol::CreateNDArray(buff,
                                             buff.CreateVector<double>(data.values()),
                                             buff.CreateVector<int32_t>(data.shape())));

    auto msg = infcomp::protocol::CreateMessage(
            buff,
            infcomp::protocol::MessageBody::ObservesInitRequest,
            observe_init.Union());

    buff.Finish(msg);
    zmq::message_t request (buff.GetSize());
    memcpy (request.data(), buff.GetBufferPointer(), buff.GetSize());
    Inference::client.send (request);
    buff.Clear();

    // TODO (Lezcano) Is this answer is unnecessary?
    zmq::message_t reply;
    Inference::client.recv (&reply);

    auto message = infcomp::protocol::GetMessage(reply.data());
    auto reply_msg = static_cast<const infcomp::protocol::ObservesInitReply*>(message->body());
    if(!reply_msg->success())
        std::cerr << "Invalid command" << std::endl;
}
}  // namespace cpprob
