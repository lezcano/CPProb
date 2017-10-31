#include <cstdlib>                                          // for exit, system
#include <iostream>                                         // for operator<<
#include <string>                                           // for operator+
#include <thread>                                           // for thread
#include <vector>                                           // for vector

#include <boost/filesystem/operations.hpp>                  // for exists
#include <boost/filesystem/path.hpp>                        // for path
#include <boost/program_options.hpp>

#include "cpprob/cpprob.hpp"                                // for compile
 // TODO(Lezcano) ciclic dependency between tratis and metapirors
// Where should tuple_observes_t be?
#include "cpprob/metapriors.hpp"                            // for tuple_obs...
#include "cpprob/postprocess/stats_printer.hpp"             // for Printer
#include "cpprob/serialization.hpp"                         // for parse_file
#include "cpprob/state.hpp"                                 // for StateType
#include "models/models.hpp"                                // for gaussian_...
#include "models/poly_adjustment.hpp"                       // for linear_re...

template <class F>
void execute (const F & f,
              const bool generate_traces, const bool compile, const bool infer, const bool sis, const bool estimate,
              const std::string & model_name,
              const std::size_t n_samples,
              const std::size_t batch_size,
              const std::string & tcp_addr_compile,
              const std::string & tcp_addr_infer,
              const std::string & nn_params) {
    using namespace boost::filesystem;
    const auto model_folder = model_name + "/";
    const auto nn_folder = model_folder + "nn";
    bool exists_nn_folder = exists(nn_folder);

    // Create folders
    if (!exists(path(model_folder))) {
        create_directory(path(model_folder));
    }
    if (!exists_nn_folder) {
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

        bool online_training = !exists(dump_folder_path);
        if (online_training) {
            std::cout << "Online Training" << std::endl;
        }
        else {
            std::cout << "Offline Training" << std::endl
                      << "Traces from Folder" << std::endl;
        }


        auto compile_command = "python3 -m pyprob.compile --batchSize " + std::to_string(batch_size) +
                                                        " --validSize " + std::to_string(batch_size) +
                                                        " --dir " + nn_folder +
                                                        " --cuda";
        if (!nn_params.empty()) {
            compile_command += " " + nn_params;
        }
        if (!online_training) {
            compile_command  += " --batchPool \"" + dump_folder + "\"";
        }
        if (exists_nn_folder) {
            compile_command += " --resume";
        }

        if (online_training) {
            std::thread thread_nn (&std::system, compile_command.c_str());

            cpprob::compile(f, tcp_addr_compile, "", 0);
            thread_nn.join();
        }
        else {
            std::system(compile_command.c_str());
        }
    }

    const std::string csis_post = model_folder + "csis.post";
    const std::string sis_post = model_folder + "sis.post";

    if (infer || sis) {
        std::cout << "Inference" << std::endl;
        const auto observes_file = model_folder + "observes.obs";

        cpprob::tuple_observes_t<F> observes;

        // TODO(Lezcano) C++17 This should be std::expected
        if (cpprob::parse_file(observes_file, observes)) {
            if (infer) {
                std::cout << "Compiled Sequential Importance Sampling (CSIS)" << std::endl;
                const cpprob::StateType state = cpprob::StateType::inference;

                auto infer_command = "python3 -m pyprob.infer --dir " + nn_folder +
                                                            " --cuda";

                if (!nn_params.empty()) {
                    infer_command += " " + nn_params;
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
        else {
            std::cout << "Error parsing " << observes_file << std::endl;
        }

    }
    if (estimate) {
        std::cout << "Posterior Distribution Estimators" << std::endl;
        const path path_dump_folder (csis_post);
        const auto print = [] (const std::string & file_name) {
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
        if (!(print(csis_post) || print(sis_post))) {
            std::cerr << "None of the files " << csis_post << " or " << sis_post << " were found." << std::endl;
        }
    }
}


int main(int argc, const char* const* argv) {

    std::size_t n_samples, batch_size;
    bool generate_traces, compile, infer, sis, estimate;
    std::string model, tcp_addr_compile, tcp_addr_infer, nn_params;
    std::vector<std::string> model_names {{"unk_mean", "unk_mean_rejection", "linear_gaussian", "hmm", "linear_regression", "unk_mean_2d", "linear_priors"}};

    std::string all_model_names = model_names[0];
    for (std::size_t i = 1; i < model_names.size(); ++i) {
        all_model_names += '/' + model_names[i];
    }

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
            ("model", po::value<std::string>(&model)->required()->value_name(all_model_names))
            ("generate_traces", po::value<bool>(&generate_traces)->default_value(false), "Generate traces for compilation.")
            ("compile", po::value<bool>(&compile)->default_value(false), "Compilation.")
            ("infer", po::value<bool>(&infer)->default_value(false), "Compiled inference.")
            ("sis", po::value<bool>(&sis)->default_value(false), "Sequential importance sampling.")
            ("estimate", po::value<bool>(&estimate)->default_value(false), "Estimators.")
            ("batch_size", po::value<std::size_t>(&batch_size)->default_value(256))
            ("n_samples", po::value<std::size_t>(&n_samples)->default_value(1000), "Number of particles to be sampled from the posterior.")
            ("tcp_addr_compile", po::value<std::string>(&tcp_addr_compile)->default_value("tcp://0.0.0.0:5555"), "Address and port to connect with the NN at compile time.")
            ("tcp_addr_infer", po::value<std::string>(&tcp_addr_infer)->default_value("tcp://127.0.0.1:6666"), "Address and port to connect with the NN at inference time.")
            ("nn_params", po::value<std::string>(&nn_params), "Parameters to send to the neural network.")
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

    auto execute_param = [&](const auto & f, const std::string & extra_params = "") {
        execute(f, generate_traces, compile, infer, sis, estimate, model, n_samples, batch_size, tcp_addr_compile, tcp_addr_infer, nn_params + extra_params);
    };

    if (model == model_names[0] /* unk_mean */) {
        execute_param(&models::gaussian_unknown_mean<>);
    }
    else if (model == model_names[1] /* unk_mean rejection */) {
        execute_param(&models::normal_rejection_sampling<>);
    }
    else if (model == model_names[2] /* linear gaussian walk */) {
        execute_param(&models::linear_gaussian_1d<50>);
    }
    else if (model == model_names[3] /* hmm */) {
        execute_param(&models::hmm<16>);
    }
    else if (model == model_names[4] /* linear regression */) {
        // Linear adjustment (Deg = 1, Points = 6)
        // execute_param(&models::poly_adjustment<1, 6>);
        execute_param(&models::linear_regression<double, 6>);
    }
    else if (model == model_names[5] /* unk_mean 2d */) {
        execute_param(&models::gaussian_2d_unk_mean<>);
    }
    else if (model == model_names[6] /* linear_priors */) {
        execute_param(&models::poly_adjustment_prior<1>, " --obsEmb lstm");
    }
    else{
        std::cerr << "Incorrect model.\n\n"
                  << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }
}
