#include <iostream>

#include <boost/program_options.hpp>

#include "models.hpp"
#include "cpprob.hpp"



template<class Func>
void train(const Func& f, const std::string& tcp_addr = "tcp://*:5556") {
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind (tcp_addr);

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
            cpprob::Core c{true};
            f(c);
            c.pack(pk);
        }

        zmq::message_t reply (sbuf.size());
        memcpy (reply.data(), sbuf.data(), sbuf.size());
        socket.send (reply);
    }
}

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    out << "[ ";
    for (const auto& elem : v)
        out << elem << " ";
    out << "]";
    return out;
}

template<typename T, typename U>
std::ostream& operator<< (std::ostream& out, const std::unordered_map<T, U>& v) {
    out << "{" << '\n';
    for (const auto& elem : v)
        out << elem.first << ": " << elem.second << '\n';
    out << "}";
    return out;
}


int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string mode;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("compile/inference"), "Compile or Inference mode");

    po::variables_map vm;
    try{
        po::store(po::parse_command_line(argc, argv, desc),  vm);
        if (vm.count("help")) {
            std::cout << "Basic Command Line Parameter App" << std::endl
                      << desc << std::endl;
            return 0;
        }

        po::notify(vm); // throws on error, so do after help in case
        // there are any problems
    }
    catch(po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    if (mode == "compile"){
        train(&mean_normal, "tcp://*:5556");
    }
    else if (mode == "inference"){
        std::cout << "Expectation example means:\n" << cpprob::expectation(&mean_normal, {0.2, 0.2}) << std::endl;
    }
    else{
        std::cout << "Incorrect mode.\n\n";
        std::cout << desc << std::endl;
    }
}
