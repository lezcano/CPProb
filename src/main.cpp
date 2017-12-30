#include <cstdlib>                                          // for exit, EXI...
#include <iostream>                                         // for operator<<
#include <string>                                           // for string
#include <tuple>                                            // for tuple
#include <utility>                                          // for make_pair

#include <boost/filesystem/operations.hpp>                  // for exists
#include <boost/filesystem/path.hpp>                        // for path
#include <boost/program_options.hpp>

#include "cpprob/call_function.hpp"                         // for call_f_de...
#include "cpprob/cpprob.hpp"                                // for compile
#include "cpprob/postprocess/stats_printer.hpp"             // for Printer
#include "cpprob/serialization.hpp"                         // for parse_file
#include "cpprob/state.hpp"                                 // for StateType

#include "models/models.hpp"
#include "models/poly_adjustment.hpp"

#ifdef BUILD_SHERPA
#include "models/sherpa.hpp"
#include "models/sherpa_mini.hpp"
#endif

namespace bf = boost::filesystem;

template<class F>
void execute(const F & model,
             const bf::path & model_folder,
             bool compile,  bool csis, bool sis, bool dryrun, bool estimate,
             const std::string & tcp_addr_compile,
             const std::string & tcp_addr_csis,
             const bool offline_training,
             const int batch_size,
             const int n_batches,
             const std::size_t n_samples,
             const bf::path & observes_file,
             const std::string & observes_str,
             const bf::path & generated_file)
{
    // Create folder
    if (!bf::exists(model_folder)) {
        bf::create_directory(model_folder);
    }

    const auto csis_post = model_folder / generated_file / "csis";
    const auto sis_post = model_folder / generated_file / "sis";

    if (compile) {
        std::cout << "Compile" << std::endl;
        if (offline_training) {
            auto dump_path = model_folder / bf::path("traces");
            std::cout << "Offline Training" << std::endl
                      << "Traces from Folder" << dump_path << std::endl;
            cpprob::compile(model, "", dump_path.string(), batch_size, n_batches);
        }
        else {
            std::cout << "Online Training" << std::endl;
            cpprob::compile(model, tcp_addr_compile, "", batch_size, n_batches);
        }
    }
    if (csis || sis) {
        // Check that exactly one of the options is set
        if (observes_file.empty() == observes_str.empty()) {
            std::cerr << R"(In CSIS or SIS mode exactly one of the options "--observes" or "--observes_file" has to be set)" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        cpprob::parameter_types_t<F, std::tuple> observes;
        bool ok;
        if (!observes_file.empty()) {
            const auto observes_path = model_folder / observes_file;
            ok = cpprob::parse_file(observes_path.string(), observes);
        }
        else {
            ok = cpprob::parse_string(observes_str, observes);
        }

        if (!ok) {
            std::cerr << "Could not parse the observations.\n"
                      << "Please use spaces to separate the observations and elements of an aggregate type instead of commas.\n"
                      << "If using the -o option, please surround the arguments by quotes.\n";
            std::exit(EXIT_FAILURE);
        }

        if (csis) {
            std::cout << "Compiled Sequential Importance Sampling (CSIS)" << std::endl;
            cpprob::generate_posterior(model, observes, tcp_addr_csis, csis_post.string(), n_samples, cpprob::StateType::csis);
        }
        if (sis) {
            std::cout << "Sequential Importance Sampling (SIS)" << std::endl;
            cpprob::generate_posterior(model, observes, "", sis_post.string(), n_samples, cpprob::StateType::sis);
        }
    }
    if (estimate) {
        std::cout << "Posterior Distribution Estimators" << std::endl;
        const auto print = [] (const bf::path & path) {
            if (bf::exists(path / bf::path("_ids"))) {
                cpprob::Printer p;
                p.load(path.string());
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
    if (dryrun) {
        std::cout << "Dry Run" << std::endl;
        cpprob::State::set(cpprob::StateType::dryrun);
        cpprob::call_f_default_params(model);
    }
}


int main(int argc, const char* const* argv) {
    namespace po = boost::program_options;

    bool compile, csis, sis, estimate, dryrun, offline_training;
    std::string model_name, model_folder, tcp_addr_compile, tcp_addr_csis, observes_file, observes_str, generated_file;
    int n_batches, batch_size;
    std::size_t n_samples;
    auto models = std::make_tuple(
            std::make_pair(std::string("unk_mean"), &models::gaussian_unknown_mean<>),
            std::make_pair(std::string("unk_mean_rejection"), &models::normal_rejection_sampling<>),
            std::make_pair(std::string("linear_gaussian"), &models::linear_gaussian_1d<50>),
            std::make_pair(std::string("hmm"), &models::hmm<4>),
            std::make_pair(std::string("linear_regression"), &models::poly_adjustment<1, 6>),
            std::make_pair(std::string("unk_mean_2d"), &models::gaussian_2d_unk_mean<>));

    // This could be generated via template meta programming...
    std::string model_names_str = std::get<0>(models).first + '/' +
                                  std::get<1>(models).first + '/' +
                                  std::get<2>(models).first + '/' +
                                  std::get<3>(models).first + '/' +
                                  std::get<4>(models).first + '/' +
                                  std::get<5>(models).first;

    #ifdef BUILD_SHERPA
    model_names_str += "/sherpa/sherpa_mini";
    #endif

    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "Print help message")
      ("compile", po::bool_switch(&compile), "Compile a NN to use in CSIS mode.")
      ("csis", po::bool_switch(&csis), "Compiled Sequential Importance Sampling: Proposals given by the NN.")
      ("sis", po::bool_switch(&sis), "Sequential Importance Sampling: Priors as proposals.")
      ("estimate", po::bool_switch(&estimate), "Estimators.")
      ("dryrun", po::bool_switch(&dryrun), "Execute the generative model.")

      ("model", po::value<std::string>(&model_name)->required()->value_name(model_names_str),
          "(Compile | CSIS | SIS | Dryrun) Select the model to be executed")
      ("model_folder", po::value<std::string>(&model_folder),
       "(Compile + --dump_to_folder | CSIS | SIS | Estimate) Model folder for the generated files.\n"
       "  Default value: Model name")

      ("tcp_addr_compile", po::value<std::string>(&tcp_addr_compile)->default_value("tcp://0.0.0.0:5555"), "(Compile) Address and port to host the trace generator server.")
      ("offline_training", po::bool_switch(&offline_training), "(Compile) Dump traces into a folder to perform offline training.")
      ("batch_size", po::value<int>(&batch_size)->default_value(128), "(Compile + --offline_training) Batch size.")
      ("n_batches", po::value<int>(&n_batches)->default_value(0), "(Compile + --offline_training) Number of batches to generate. If equal to 0 it will generate batches forever.")

      ("tcp_addr_infer", po::value<std::string>(&tcp_addr_csis)->default_value("tcp://127.0.0.1:6666"), "(CSIS) Address and port to connect to the NN.")
      ("n_samples,n", po::value<std::size_t>(&n_samples)->default_value(10000), "(CSIS | SIS) Number of particles to be sampled from the posterior.")
      ("observes,o", po::value<std::string>(&observes_str), "(CSIS | SIS) Values to observe.")
      ("observes_file,f", po::value<std::string>(&observes_file), "(CSIS | SIS) File with the observed values.")
      ("generated_file", po::value<std::string>(&generated_file)->default_value(""), "(CSIS | SIS | Estimate) File for the samples from the posterior.")
      ;
    #ifdef BUILD_SHERPA
    std::cout << "SHERPA model built." << std::endl;
    #endif

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc),  vm);
        if (vm.count("help") != 0u) {
            std::cout << "CPProb Inference Library" << std::endl
                      << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    }
    catch(po::error& e) {
        std::cerr << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        std::exit (EXIT_FAILURE);
    }

    // Set model_folder to default
    if (model_folder.empty()) {
        model_folder = model_name;
    }

    const auto execute_params = [&](const auto & model) {
        execute(model, bf::path(model_folder),
                compile, csis, sis, dryrun, estimate,
                tcp_addr_compile, tcp_addr_csis, offline_training, batch_size,
                n_batches, n_samples, bf::path(observes_file), observes_str, bf::path(generated_file));
    };

    // This could be generated via template meta programming...
    if (model_name == std::get<0>(models).first) {
        execute_params(std::get<0>(models).second);
    }
    else if (model_name == std::get<1>(models).first) {
        execute_params(std::get<1>(models).second);
    }
    else if (model_name == std::get<2>(models).first) {
        execute_params(std::get<2>(models).second);
    }
    else if (model_name == std::get<3>(models).first) {
        execute_params(std::get<3>(models).second);
    }
    else if (model_name == std::get<4>(models).first) {
        execute_params(std::get<4>(models).second);
    }
    else if (model_name == std::get<5>(models).first) {
        execute_params(std::get<5>(models).second);
    }
    #ifdef BUILD_SHERPA
    else if (model_name  == "sherpa") {
        execute_params(models::SherpaWrapper{});
    }
    else if (model_name  == "sherpa_mini") {
        execute_params(&models::sherpa_mini_wrapper);
    }
    #endif
    else {
        std::cerr << "Model not available. Please provide one of the following:" << std::endl
                  << model_names_str << std::endl;
        std::exit (EXIT_FAILURE);
    }
}
