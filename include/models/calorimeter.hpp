#ifndef CPPROB_CALORIMETER_HPP
#define CPPROB_CALORIMETER_HPP

#include <vector>
#include <boost/random/uniform_real.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <algorithm>
#include <Rivet/Math/Matrix3.hh>
#include "Rivet/Tools/ParticleIdUtils.hh"
#include <cpprob/cpprob.hpp>
#include <cpprob/distributions/multivariate_normal.hpp>

namespace models {

std::vector<std::vector<std::vector<double> > > calo_simulation(const std::vector<std::vector<double> > &particle_data);

} // namespace models

#endif //CPPROB_CALORIMETER_HPP
