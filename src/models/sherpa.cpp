#include "models/sherpa.hpp"

#include <string>
#include <vector>
#include <iostream>     // std::cout, std::ostream, std::ios
#include <exception>    // std::terminate


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

namespace sherpa_detail {


void sherpa_wrapper(const std::vector<std::vector<std::vector<double>>> &observes)
{
    cpprob::predict(jailbreak::instance().m_selected_channel_index);
    const double OBS_WIDTH = 1.0;
    auto sherpa_img = sherpa();
    cpprob::multivariate_normal_distribution<double> obs_distr(cpprob::NDArray<double>(sherpa_img), OBS_WIDTH);
    cpprob::observe(obs_distr, observes);
}


std::vector<std::vector<std::vector<double>>> sherpa()
{
    jailbreak::instance().m_histo3d.clear();
    ::SHERPA::Sherpa Generator{};
    try {
        int sherpa_argc = 5;
        char *sherpa_argv[] = {"some_binary", "-f", "Gun.dat", "EXTERNAL_RNG=ProbProbRNG",
                               "SHERPA_LDADD=ProbProgRNG"};
        Generator.InitializeTheRun(sherpa_argc, sherpa_argv);
        Generator.InitializeTheEventHandler();
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }

    try {
        while (!Generator.GenerateOneEvent());
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }

    std::cout << "SHERPAPROBPROG: successfully generated an event!!" << std::endl;
    std::cout << "SHERPAPROBPROG: jailbroken value is: " << jailbreak::instance().m_histo3d.size() << " ... "
              << std::endl;
    std::cout << "----" << std::endl;

    auto ret = jailbreak::instance().m_histo3d;

    Generator.SummarizeRun();
    return ret;
}
} // end namespace sherpa_details
