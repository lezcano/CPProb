#include <iostream>
#include <tuple>

#include <boost/function_types/parameter_types.hpp>
#include <boost/program_options.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/traits.hpp"

#include "models/models.hpp"
#include "models/sherpa_mini.hpp"

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[ ";
    for (const auto &elem : v) {
        out << elem << " ";
    }
    out << "]";
    return out;
}

int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string mode, tcp_addr, observes_file, observes_str, model_file;
    size_t n_samples;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help messages")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("dryrun/compile/infer"), "Compile, Inference or Dry Run mode.")
      ("n_samples,n", po::value<size_t>(&n_samples)->default_value(10000), "(Inference) Number of particles to be sampled.")
      ("tcp_addr,a", po::value<std::string>(&tcp_addr), "Address and port to connect with the rnn.\n"
                                                        "Defaults:\n"
                                                        "  Compile:   tcp://0.0.0.0:5555\n"
                                                        "  Inference: tcp://127.0.0.1:6666\n"
                                                        "  Dry Run:   None")
      ("observes,o", po::value<std::string>(&observes_str), "(Inference) Values to observe.")
      ("observes_file,f", po::value<std::string>(&observes_file), "(Inference) File with the observed values.")
      ("model_file", po::value<std::string>(&model_file), "(Inference) File to output the posterior distribution.")
      ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc),  vm);
        if (vm.count("help") != 0u) {
            std::cout << "CPProb Compiled Inference Library" << std::endl
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

    auto f = &cpprob::models::sherpa_wrapper;

    if (mode == "compile"){
        if (tcp_addr.empty()) {
            tcp_addr = "tcp://0.0.0.0:5555";
        }
        cpprob::compile(f, tcp_addr);
    }
    else if (mode == "infer"){
        if (tcp_addr.empty()) {
            tcp_addr = "tcp://127.0.0.1:6666";
        }
        // Check that exactly one of the options is set
        if (vm.count("observes") == vm.count("observes_file")) {
            std::cerr << R"(In Infer mode exactly one of the options "--observes" or "--observes_file" has to be set)" << std::endl;
            return -1;
        }

        if (vm.count("model_file") == 0u) {
            std::cerr << R"(In Infer mode, please provide a "--model_file" argument.)" << std::endl;
            return -1;
        }

        // The return type of parse_file_param_f and parse_string_param_f is the same
        using tuple_params_t = cpprob::parameter_types_t<decltype(f), std::tuple>;

        tuple_params_t observes;
        bool ok;
        if (vm.count("observes_file") != 0u) {
            ok = cpprob::parse_file(observes_file, observes);
        }
        else {
            ok = cpprob::parse_string(observes_str, observes);
        }
        if (ok) {
            cpprob::generate_posterior(f, observes, tcp_addr, model_file, n_samples);
        }
        else {
            std::cerr << "Could not parse the arguments.\n"
                      << "Use spaces to separate the arguments and elements of an aggregate type.\n";
        }
    }
    else{
        std::cerr << "Incorrect mode.\n\n"
                  << desc << std::endl;
    }
}
