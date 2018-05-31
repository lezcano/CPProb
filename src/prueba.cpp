#include <iostream>
#include <string>
#include <tuple>
#include "models/mod_prueba.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/postprocess/stats_printer.hpp"
int main () {
    const auto observes = std::make_tuple(3., 4.);
    const auto samples = 10'000;
    const std::string outfile = "posterior_sis";
    cpprob::inference(cpprob::StateType::sis, &models::gaussian_unknown_mean, observes, samples, outfile);
    std::cout << cpprob::StatsPrinter{outfile} << std::endl;
}
