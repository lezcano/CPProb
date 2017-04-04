#include "cpprob/socket.hpp"

#include <iostream>
#include <string>

#include <zmq.hpp>
#include <msgpack.hpp>

#include "cpprob/trace.hpp"

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
/////////////////////////          Inference            ////////////////////////
////////////////////////////////////////////////////////////////////////////////
zmq::context_t context (1);
zmq::socket_t client (context, ZMQ_REQ);

void connect_client(const std::string& tcp_addr){
    client.connect (tcp_addr);
}

void send_observe_init(std::vector<double>&& data){
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
    client.send (request);

    // TODO (Lezcano) This answer is unnecessary
    client.recv (&reply);
    auto rpl = std::string(static_cast<char*>(reply.data()), reply.size());

    std::string answer = msgpack::unpack(rpl.data(), rpl.size()).get().as<std::string>();
    if(answer != "observe-received")
        std::cout << "Invalid command " << answer << std::endl;
}

std::vector<double> get_params(const SampleInference& curr_sample, const PrevSampleInference& prev_sample){
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(&sbuf);

    pk.pack_map(2);
    pk.pack(std::string("command"));
    pk.pack(std::string("proposal-params"));

    pk.pack(std::string("command-param"));
    pk.pack_map(6);

    pk.pack(std::string("sample-address"));
    pk.pack(curr_sample.sample_address);

    pk.pack(std::string("sample-instance"));
    pk.pack(curr_sample.sample_instance);

    pk.pack(std::string("proposal-name"));
    pk.pack(std::string(curr_sample.proposal_name));

    pk.pack(std::string("prev-sample-address"));
    pk.pack(prev_sample.prev_sample_address);

    pk.pack(std::string("prev-sample-instance"));
    pk.pack(prev_sample.prev_sample_instance);

    pk.pack(std::string("prev-sample-value"));
    pk.pack(prev_sample.prev_sample_value);


    zmq::message_t request (sbuf.size());
    memcpy (request.data(), sbuf.data(), sbuf.size());
    client.send(request);

    return receive<std::vector<double>>(client);
}



////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

zmq::socket_t server (context, ZMQ_REP);
int batch_size = 0;
msgpack::sbuffer sbuf;

void connect_server(const std::string& tcp_addr){
    server.bind (tcp_addr);
}

void add_trace(const Trace& t, int num){
    static msgpack::packer<msgpack::sbuffer> pk(&sbuf);
    if (num == 0){
        pk.pack_array(batch_size);
    }
    t.pack(pk);
}

int get_batch_size(){
    zmq::message_t request;

    server.recv (&request);
    // oh is necessary!!
    auto oh = msgpack::unpack(static_cast<char*>(request.data()), request.size());
    msgpack::object obj = oh.get();
    auto pkg = obj.as<std::map<std::string, msgpack::object>>();
    auto command = pkg["command"].as<std::string>();
    if (command == "new-batch"){
        batch_size = pkg["command-param"].as<int>();
        return batch_size;
    }
    else{
        std::cout << "Invalid command " << pkg["command"].as<std::string>() << std::endl;
        return 0;
    }
}

void send_batch(){
    zmq::message_t reply (sbuf.size());
    memcpy (reply.data(), sbuf.data(), sbuf.size());
    server.send (reply);
    sbuf.clear();
}

}  // namespace cpprob
