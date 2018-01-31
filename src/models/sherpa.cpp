#include "models/sherpa.hpp"

#include <string>
#include <vector>
#include <iostream>     // std::cout, std::ostream, std::ios
#include <exception>    // std::terminate
#include <memory>
#include <tuple>


#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"
#include "ATOOLS/Org/AnalysisJailbreak.H"
#include "models/calorimeter.hpp"

#include "cpprob/cpprob.hpp"
#include "cpprob/distributions/abc.hpp"
#include "cpprob/distributions/dirac_delta.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/ndarray.hpp"

namespace models {

SherpaWrapper::SherpaWrapper() : generator_{new ::SHERPA::Sherpa}
{
    try {
        const int sherpa_argc = 7;
        char* sherpa_argv[] = {"some_binary","-f","Gun.dat","EXTERNAL_RNG=ProbProbRNG","SHERPA_LDADD=ProbProgRNG","OUTPUT=0","LOG_FILE=/dev/null"};
        generator_->InitializeTheRun(sherpa_argc, sherpa_argv);
        generator_->InitializeTheEventHandler();
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }
}

SherpaWrapper::~SherpaWrapper()
{
    generator_->SummarizeRun();
    delete generator_;
}


void SherpaWrapper::operator()(const std::vector<std::vector<std::vector<double>>> &observes) const
{
    constexpr double variance = 1e-2; // The covariance matrix is 1e-4 Id
    int channel_index;
    std::vector<double> mother_momentum;
    std::vector<std::vector<double>> final_state_particles;

    std::tie(channel_index, mother_momentum, final_state_particles) = sherpa();
    cpprob::NDArray<double> calo_histo = calo_simulation(final_state_particles);

    auto dirac = cpprob::make_dirac_delta(calo_histo);
    // Use \sqrt(calo_histo) as the standard deviation for th emultivariate normal
    cpprob::multivariate_normal_distribution<double> approximate_dirac(calo_histo, calo_histo);
    cpprob::observe(cpprob::make_abc(dirac, approximate_dirac), observes);
    cpprob::predict(channel_index, "Decay Channel");
    cpprob::predict(mother_momentum[0], "Momentum X");
    cpprob::predict(mother_momentum[1], "Momentum Y");
    cpprob::predict(mother_momentum[2], "Momentum Z");
}

std::tuple<int,
           std::vector<double>,
           std::vector<std::vector<double>>>
SherpaWrapper::sherpa() const
{
    try {
        while (!generator_->GenerateOneEvent());
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }

    return std::make_tuple(jailbreak::instance().m_selected_channel_index,
                           jailbreak::instance().m_mother_momentum,
                           jailbreak::instance().m_final_state_particles);
}

} // end namespace models
