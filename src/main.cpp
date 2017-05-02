#include <iostream>

#include <boost/program_options.hpp>
#include <boost/function_types/parameter_types.hpp>

#include "cpprob/serialization.hpp"
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

    std::string mode, tcp_addr, observes_file, observes_str;
    int n_samples;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("compile/infer")->default_value("compile"), "Compile or Inference mode.")
      ("n_samples,n", po::value<int>(&n_samples)->default_value(10000), "Number of particles to be sampled during inference.")
      ("tcp_addr,a", po::value<std::string>(&tcp_addr), "Address and port to connect with the rnn. Default 127.0.0.1 and port 5555 for compile, 6666 for inference.")
      ("observes,o", po::value<std::string>(&observes_str), "Values to observe. Used in Inference mode.")
      ("observes_file,f", po::value<std::string>(&observes_file), "File with the observed values in Serialized format. Used in Inference mode.")
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

    auto f = &cpprob::models::least_sqr;

    if (mode == "compile"){
        if (tcp_addr.empty())
            tcp_addr = "tcp://*:5555";
        cpprob::compile(f, tcp_addr);
    }
    else if (mode == "infer"){
        if (tcp_addr.empty())
            tcp_addr = "tcp://localhost:6666";
        // Check that exactly one of the options is set
        if (vm.count("observes") == vm.count("observes_file")) {
            std::cerr << "Exactly one of the options \"--observes\" or \"--observes_file\" has to be set" << std::endl;
            return -1;
        }
        decltype(cpprob::load_csv_f<decltype(f)>(observes_file)) observes;
        if (vm.count("observes_file"))
            observes = cpprob::load_csv_f<decltype(f)>(observes_file);
        else
            observes = cpprob::parse_param_f<decltype(f)>(observes_str);
        std::cout << std::get<0>(observes).size() << std::endl;
        std::cout << "Expectation example means:\n" << cpprob::inference(f, tcp_addr, n_samples, observes) << std::endl;
    }
    else{
        std::cerr << "Incorrect mode.\n\n";
        std::cerr << desc << std::endl;
    }
}
