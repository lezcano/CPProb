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

#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/ndarray.hpp"

namespace models {

SherpaWrapper::SherpaWrapper() : generator_{new ::SHERPA::Sherpa}
{
    jailbreak::instance().m_histo3d.clear();
    try {
        const int sherpa_argc = 7;
        const char* sherpa_argv[] = {"some_binary","-f","Gun.dat","EXTERNAL_RNG=ProbProbRNG","SHERPA_LDADD=ProbProgRNG","OUTPUT=0","LOG_FILE=/dev/null"};
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
    const double OBS_WIDTH = 1.0;
    int channel_index;
    std::vector<double> mother_momentum;
    std::vector<std::vector<std::vector<double>>> img;

    std::tie(channel_index, mother_momentum, img) = sherpa_pred_obs();

    cpprob::multivariate_normal_distribution<double> likelihood(cpprob::NDArray<double>(img), OBS_WIDTH);
    cpprob::observe(likelihood, observes);
    cpprob::predict(channel_index);
    cpprob::predict(mother_momentum);
}

std::tuple<int,
           std::vector<double>,
           std::vector<std::vector<std::vector<double>>>>
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
                           jailbreak::instance().m_histo3d);
}

} // end namespace models
