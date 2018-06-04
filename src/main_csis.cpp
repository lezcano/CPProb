#include <iostream>
#include <string>
#include <tuple>
#include "models/gaussian.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/postprocess/stats_printer.hpp"

int main (int argc, char* argv[]) {
    if (argc != 2) { std::cout << "No arguments provided.\n"; return 1; }
    if (argv[1] == std::string("compile")) {
        cpprob::compile(&models::gaussian_unknown_mean);
    }
    else if (argv[1] == std::string("infer")) {
        const auto observes = std::make_tuple(3., 4.);
        const auto samples  = 100;
        const auto outfile  = std::string("posterior_csis");
        cpprob::inference(cpprob::StateType::csis, &models::gaussian_unknown_mean, observes, samples, outfile);
        std::cout << cpprob::StatsPrinter{outfile} << std::endl;
    }
}
