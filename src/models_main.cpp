#include <algorithm>
#include <cstdlib>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/traits.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

#include "models/models.hpp"
#include "models/poly_adjustment.hpp"

template <class F>
void execute (const F & f,
              const bool generate_traces, const bool compile, const bool infer, const bool sis, const bool estimate,
              const std::string & model_name,
              const std::size_t n_samples,
              const std::size_t batch_size,
              const std::string & tcp_addr_compile,
              const std::string & tcp_addr_infer,
              const bool optirun) {
    using namespace boost::filesystem;
    const auto model_folder = model_name + "/";
    const auto nn_folder = model_folder + "nn";

    // Create folders
    if (!exists(path(model_folder))) {
        create_directory(path(model_folder));
    }
    if (!exists(nn_folder)) {
        create_directory(path(nn_folder));
    }

    const auto dump_folder = model_folder + "traces";
    const path dump_folder_path{dump_folder};

    if (generate_traces) {
        std::cout << "Generating Traces" << std::endl;
        if (!exists(dump_folder_path)) {
            create_directory(path(dump_folder_path));
        }
        cpprob::compile(f, tcp_addr_compile, dump_folder, batch_size);
    }

    if (compile) {
        std::cout << "Compile" << std::endl;

        if (!exists(dump_folder_path)) {
            std::cout << "Traces folder does not exist. Could not compile the NN." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        auto compile_command = "python3 -m infcomp.compile --batchPool \"" + dump_folder + "\"" +
                                                         " --batchSize " + std::to_string(batch_size) +
                                                         " --validSize " + std::to_string(batch_size) +
                                                         " --dir " + nn_folder +
                                                         " --cuda";
        if (optirun) {
            compile_command = "optirun " + compile_command;
        }

        std::system(compile_command.c_str());
    }

    const std::string csis_post = model_folder + "csis.post";
    const std::string sis_post = model_folder + "sis.post";

    if (infer || sis) {
        std::cout << "Inference" << std::endl;
        const auto observes_file = model_folder + "observes.obs";

        using tuple_params_t = cpprob::parameter_types_t<F, std::tuple>;
        tuple_params_t observes;

        if (cpprob::parse_file(observes_file, observes)) {
            if (infer) {
                std::cout << "Compiled Sequential Importance Sampling (CSIS)" << std::endl;
                const cpprob::StateType state = cpprob::StateType::inference;

                auto infer_command = "python3 -m infcomp.infer --dir " + nn_folder +
                                                             " --cuda";

                if (optirun) {
                    infer_command = "optirun " + infer_command;
                }

                std::thread thread_nn (&std::system, infer_command.c_str());

                cpprob::generate_posterior(f, observes, tcp_addr_infer, csis_post, n_samples, state);
                thread_nn.join();
            }
            if (sis) {
                std::cout << "Sequential Importance Sampling (SIS)" << std::endl;
                const cpprob::StateType state = cpprob::StateType::importance_sampling;
                cpprob::generate_posterior(f, observes, tcp_addr_infer, sis_post, n_samples, state);
            }
        }

    }
    if (estimate) {
        std::cout << "Posterior Distribution Estimators" << std::endl;
        const path path_dump_folder (csis_post);
        auto print = [] (const std::string & file_name) {
            if (exists(path(file_name + "_ids"))) {
                cpprob::Printer p;
                p.load(file_name);
                p.print(std::cout);
                return true;
            }
            else {
                return false;
            }
        };
        bool one_exists = false;
        one_exists |= print(csis_post);
        one_exists |= print(sis_post);
        if (!one_exists) {
            std::cerr << "None of the files " << csis_post << " or " << sis_post << " were found.";
        }
    }
}


int main(int argc, const char* const* argv) {

    std::size_t n_samples, batch_size;
    bool generate_traces, compile, infer, sis, estimate, all, optirun;
    std::string model, tcp_addr_compile, tcp_addr_infer;
    std::vector<std::string> model_names {{"unk_mean", "unk_mean_rejection", "linear_gaussian", "hmm", "linear_regression", "unk_mean_2d"}};

    std::string all_model_names = model_names[0];
    for (std::size_t i = 1; i < model_names.size(); ++i) {
        all_model_names += '/' + model_names[i];
    }

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
            ("model", po::value<std::string>(&model)->required()->value_name(all_model_names))
            ("generate_traces", "Generate traces for compilation.")
            ("compile", "Execute compilation.")
            ("infer", "Execute compiled inference.")
            ("sis", "Execute sequential importance sampling.")
            ("estimate", "Execute estimators.")
            ("all", "Execute all the options")
            ("batch_size", po::value<std::size_t>(&batch_size)->default_value(256))
            ("n_samples", po::value<std::size_t>(&n_samples)->default_value(1000), "Number of particles to be sampled from the posterior.")
            ("tcp_addr_compile", po::value<std::string>(&tcp_addr_compile)->default_value("tcp://0.0.0.0:5555"), "Address and port to connect with the NN at compile time.")
            ("tcp_addr_infer", po::value<std::string>(&tcp_addr_infer)->default_value("tcp://127.0.0.1:6666"), "Address and port to connect with the NN at inference time.")
            ("optirun", "Execute python with optirun")
            ;

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc),  vm);
        if (vm.count("help") != 0u) {
            std::cout << "CPProb Models" << std::endl
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
    generate_traces = vm.count("generate_traces");
    compile = vm.count("compile");
    infer = vm.count("infer");
    sis = vm.count("sis");
    estimate = vm.count("estimate");
    all = vm.count("all");
    optirun = vm.count("optirun");

    if (all && (compile || infer || sis || estimate)) {
        std::cerr << "Select all or some parts of the compilation, but not both."
                  << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }
    if (all) {
        generate_traces = true;
        compile = true;
        infer = true;
        sis = true;
        estimate = true;
    }

    if (model == model_names[0] /* unk_mean */) {
        auto f = &models::gaussian_unknown_mean<>;
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else if (model == model_names[1] /* unk_mean rejection */) {
        auto f = &models::normal_rejection_sampling<>;
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else if (model == model_names[2] /* linear gaussian walk */) {
        auto f = &models::linear_gaussian_1d<50>;
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else if (model == model_names[3] /* hmm */) {
        auto f = &models::hmm<16>;
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else if (model == model_names[4] /* linear regression */) {
        auto f = &models::poly_adjustment<1, 6>; // Linear adjustment (Deg = 1, Points = 6)
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else if (model == model_names[5] /* unk_mean 2d */) {
        auto f = &models::gaussian_2d_unk_mean<>; // Linear adjustment (Deg = 1, Points = 6)
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, optirun);
    }
    else{
        std::cerr << "Incorrect model.\n\n"
                  << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }
}
