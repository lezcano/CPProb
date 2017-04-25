#include <iostream>

#include <boost/program_options.hpp>

#include "models/models.hpp"
#include "cpprob/cpprob.hpp"

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[ ";
    for (const auto &elem : v)
        out << elem << " ";
    out << "]";
    return out;
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string mode, tcp_addr;
    int n_samples;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("compile/inference")->default_value("compile"), "Compile or Inference mode")
      ("n_samples,n", po::value<int>(&n_samples)->default_value(10000), "Number of particles to be sampled during inference")
      ("tcp_addr,a", po::value<std::string>(&tcp_addr), "Address and port to connect with the rnn. Default 127.0.0.1 and port 5555 for compile, 6666 for inference.")
      ;

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

    auto all_distr = [](const int y1 = 0, const int y2 = 0){return cpprob::models::all_distr(y1, y2);};

    if (mode == "compile"){
        if (tcp_addr.empty())
            tcp_addr = "tcp://*:5555";
        cpprob::compile(all_distr, tcp_addr);
    }
    else if (mode == "inference"){
        if (tcp_addr.empty())
            tcp_addr = "tcp://localhost:6666";
        std::cout << "Expectation example means:\n" << cpprob::inference(all_distr, tcp_addr, n_samples, 3, 4) << std::endl;
    }
    else{
        std::cout << "Incorrect mode.\n\n";
        std::cout << desc << std::endl;
    }
}
