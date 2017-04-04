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

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("mode,m", po::value<std::string>()->required()->value_name("compile/inference"), "Compile or Inference mode");

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

    auto mode = vm["mode"].as<std::string>();

    if (mode == "compile"){
        std::string tcp_addr = "tcp://*:5556";
        cpprob::compile(&mean_normal, tcp_addr);
    }
    else if (mode == "inference"){
        std::cout << "Expectation example means:\n" << cpprob::inference(&mean_normal, {0.2, 0.2}) << std::endl;
    }
    else{
        std::cout << "Incorrect mode.\n\n";
        std::cout << desc << std::endl;
    }
}
