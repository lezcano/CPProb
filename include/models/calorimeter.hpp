#ifndef CPPROB_CALORIMETER_HPP
#define CPPROB_CALORIMETER_HPP

#include <vector>

namespace models {

std::vector<std::vector<std::vector<double>>> calo_simulation(const std::vector<std::vector<double>> &particle_data);

}

#endif //CPPROB_CALORIMETER_HPP
