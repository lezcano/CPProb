#include <iostream>
#include <tuple>
#include <cstdlib>

#include <boost/function_types/parameter_types.hpp>
#include <boost/program_options.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/traits.hpp"

#include "models/models.hpp"
#include "models/sherpa_mini.hpp"
#include "models/sherpa.hpp"

template <class T>
bool get_observes(const boost::program_options::variables_map & vm,
                  T& observes,
                  const std::string & observes_file,
                  const std::string & observes_str) {
    // Check that exactly one of the options is set
    if (vm.count("observes") == vm.count("observes_file")) {
        std::cerr << R"(In Infer mode exactly one of the options "--observes" or "--observes_file" has to be set)" << std::endl;
        std::exit (EXIT_FAILURE);
    }

    if (vm.count("posterior_file") == 0u) {
        std::cerr << R"(In Infer mode, please provide a "--posterior_file" argument.)" << std::endl;
        std::exit (EXIT_FAILURE);
    }

    if (vm.count("observes_file") != 0u) {
        return cpprob::parse_file(observes_file, observes);
    }
    else {
        return cpprob::parse_string(observes_str, observes);
    }
}


int main(int argc, char** argv) {
    namespace po = boost::program_options;

    std::string mode, tcp_addr, observes_file, observes_str, posterior_file;
    size_t n_samples;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help message")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("compile/infer/dryrun/infer-regular"), "Compile, Inference, Dry Run or Importance Sampling with Priors as Proposals mode.")
      ("n_samples,n", po::value<size_t>(&n_samples)->default_value(10000), "(Inference) Number of particles to be sampled.")
      ("tcp_addr,a", po::value<std::string>(&tcp_addr), "Address and port to connect with the rnn.\n"
                                                        "Defaults:\n"
                                                        "  Compile:   tcp://0.0.0.0:5555\n"
                                                        "  Inference: tcp://127.0.0.1:6666\n"
                                                        "  Dry Run:   None"
                                                        "  Regular:   None")
      ("observes,o", po::value<std::string>(&observes_str), "(Inference | Importance Sampling) Values to observe.")
      ("observes_file,f", po::value<std::string>(&observes_file), "(Inference | Regular) File with the observed values.")
      ("posterior_file", po::value<std::string>(&posterior_file), "(Inference | Regular) File to output the posterior distribution.")
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
        std::exit (EXIT_FAILURE);
    }

    #ifdef BUILD_SHERPA
    sherpa_detail::SherpaWrapper f{};
    #else
    auto f = &cpprob::models::sherpa_wrapper;
    #endif

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

        // The return type of parse_file_param_f and parse_string_param_f is the same
        using tuple_params_t = cpprob::parameter_types_t<decltype(f), std::tuple>;

        tuple_params_t observes;
        if (get_observes(vm, observes, observes_file, observes_str)) {
            cpprob::generate_posterior(f, observes, tcp_addr, posterior_file, n_samples);
        }
        else {
            std::cerr << "Could not parse the arguments.\n"
                      << "Use spaces to separate the arguments and elements of an aggregate type.\n";
            std::exit (EXIT_FAILURE);
        }

    }
    else if (mode == "infer-regular") {
        // The return type of parse_file_param_f and parse_string_param_f is the same
        using tuple_params_t = cpprob::parameter_types_t<decltype(f), std::tuple>;

        tuple_params_t observes;
        if (get_observes(vm, observes, observes_file, observes_str)) {
            cpprob::importance_sampling(f, observes, posterior_file, n_samples);
        }
        else {
            std::cerr << "Could not parse the arguments.\n"
                      << "Use spaces to separate the arguments and elements of an aggregate type.\n";
            std::exit (EXIT_FAILURE);
        }

    }
    else{
        std::cerr << "Incorrect mode.\n\n"
                  << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }
}
