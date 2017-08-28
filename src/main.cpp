#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tuple>

#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/state.hpp"
#include "cpprob/call_function.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

#ifdef BUILD_SHERPA
#include "models/sherpa.hpp"
#include "models/sherpa_mini.hpp"
#else
#include "models/models.hpp"
#include "models/poly_adjustment.hpp"
#endif

template <class... T>
bool get_observes(const boost::program_options::variables_map & vm,
                  std::tuple<T...>& observes,
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


int main(int argc, const char* const* argv) {
    namespace po = boost::program_options;

    std::string mode, tcp_addr, observes_file, observes_str, posterior_file, dump_folder;
    std::size_t n_samples;

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help message")
      ("mode,m", po::value<std::string>(&mode)->required()->value_name("compile/infer/infer-regular/estimate/dryrun"),
                                                                       "Compile: Compile a NN to use in inference mode.\n"
                                                                       "Infer: Perform compiled inference.\n"
                                                                       "Infer-Regular: Perform sequential importance sampling with priors as proposals.\n"
                                                                       "Estimate: Compute estimators of the posterior distribution.\n"
                                                                       "Dry Run: Execute the model without any inference algorithm.")
      ("n_samples,n", po::value<std::size_t>(&n_samples)->default_value(10000), "(Compile + --dump_folder | Inference) Number of particles to be sampled.")
      ("tcp_addr,a", po::value<std::string>(&tcp_addr), "Address and port to connect with the NN.\n"
                                                        "Defaults:\n"
                                                        "  Compile:   tcp://0.0.0.0:5555\n"
                                                        "  Inference: tcp://127.0.0.1:6666")
      ("dump_folder", po::value<std::string>(&dump_folder), "(Compilation) Dump traces into a file.")
      ("observes,o", po::value<std::string>(&observes_str), "(Inference | Regular) Values to observe.")
      ("observes_file,f", po::value<std::string>(&observes_file), "(Inference | Regular) File with the observed values.")
      ("posterior_file,p", po::value<std::string>(&posterior_file), "(Inference | Regular | Estimate) Posterior distribution file.")
      ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc),  vm);
        if (vm.count("help") != 0u) {
            std::cout << "CPProb Inference Library" << std::endl
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
    models::SherpaWrapper f;
    // auto f = &sherpa_mini_wrapper;
    #else
    // auto f = &models::normal_rejection_sampling<>;
    // auto f = &models::linear_gaussian_1d<50>;
    // auto f = &models::gaussian_unknown_mean<>;
    // auto f = &models::hmm<16>;
    // auto f = &models::poly_adjustment<1, 6>; // Linear adjustment (Deg = 1, Points = 6)
    models::Gauss<> f{};
    #endif

    if (mode == "compile") {
        if (tcp_addr.empty()) {
            tcp_addr = "tcp://0.0.0.0:5555";
        }

        if (vm.count("dump_folder") != 0u) {
            const boost::filesystem::path path_dump_folder(dump_folder);
            if (!boost::filesystem::exists(path_dump_folder)) {
                std::cerr << "Provided --dump_folder \"" + dump_folder + "\" does not exist.\n"
                          << "Please provide a valid folder.\n";
                std::exit(EXIT_FAILURE);
            }
        }

        cpprob::compile(f, tcp_addr, dump_folder, n_samples);
    }
    else if (mode == "infer" || mode == "infer-regular") {
        if (tcp_addr.empty()) {
            tcp_addr = "tcp://127.0.0.1:6666";
        }

        using tuple_params_t = cpprob::parameter_types_t<decltype(f), std::tuple>;
        tuple_params_t observes;

        if (get_observes(vm, observes, observes_file, observes_str)) {
            cpprob::StateType state =
                    [](const std::string & mode) {
                        if (mode == "infer") {
                            return cpprob::StateType::inference;
                        }
                        else if (mode == "infer-regular") {
                            return cpprob::StateType::importance_sampling;
                        }
                        else {
                            std::exit (EXIT_FAILURE);
                        }
                    }(mode);
            cpprob::generate_posterior(f, observes, tcp_addr, posterior_file, n_samples, state);
        }
        else {
            std::cerr << "Could not parse the arguments.\n"
                      << "Please use spaces to separate the arguments and elements of an aggregate type.\n"
                      << "If using the -o option, please surround the arguments by quotes.\n";
            std::exit (EXIT_FAILURE);
        }

    }
    else if (mode == "estimate") {
        cpprob::Printer p;
        p.load(posterior_file);
        p.print(std::cout);
    }
    else if (mode == "dryrun") {
        cpprob::State::set(cpprob::StateType::dryrun);
        cpprob::call_f_default_params(f);
    }
    else{
        std::cerr << "Incorrect mode.\n\n"
                  << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }
}
