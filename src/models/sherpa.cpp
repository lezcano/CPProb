#include "models/sherpa.hpp"

#include <string>
#include <vector>
#include <iostream>     // std::cout, std::ostream, std::ios
#include <exception>    // std::terminate
#include <memory>


#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"
#include "ATOOLS/Org/AnalysisJailbreak.H"

#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/cpprob.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {
namespace models {

SherpaWrapper::SherpaWrapper() : generator_{new SHERPA::Sherpa}
{
    jailbreak::instance().m_histo3d.clear();
    try {
        int sherpa_argc = 7;
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
    delete generator_;
}


void SherpaWrapper::operator()(const std::vector<std::vector<std::vector<double>>> &observes) const
{
    cpprob::predict(jailbreak::instance().m_selected_channel_index);
    cpprob::predict(jailbreak::instance().m_mother_momentum);
    const double OBS_WIDTH = 1.0;
    auto sherpa_img = this->sherpa();
    cpprob::multivariate_normal_distribution<double> obs_distr(cpprob::NDArray<double>(sherpa_img), OBS_WIDTH);
    cpprob::observe(obs_distr, observes);
}


std::vector<std::vector<std::vector<double>>> SherpaWrapper::sherpa()
{
    try {
        while (!generator_->GenerateOneEvent());
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }

    std::cout << "SHERPAPROBPROG: successfully generated an event!!" << std::endl;
    std::cout << "SHERPAPROBPROG: Selected Channel Index: " << jailbreak::instance().m_selected_channel_index << std::endl;
    std::cout << "SHERPAPROBPROG: Mother Momentum: " << jailbreak::instance().m_mother_momentum << std::endl;
    std::cout << "SHERPAPROBPROG: jailbroken value is: " << jailbreak::instance().m_histo3d.size() << " ... "
              << std::endl;
    std::cout << "----" << std::endl;

    auto ret = jailbreak::instance().m_histo3d;

    generator_->SummarizeRun();
    return ret;
}
} // end namespace models
} // end namespace cpprob
