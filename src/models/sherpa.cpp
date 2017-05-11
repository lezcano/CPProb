#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"
#include "ATOOLS/Org/AnalysisJailbreak.H"

using namespace SHERPA;
using namespace ATOOLS;

#include <iostream>     // std::cout, std::ostream, std::ios
#include <exception>    // std::terminate


std::vector<std::vector<std::vector<double>>> sherpa_wrapper()
{
    jailbreak::instance().m_histo3d.clear();
    static Sherpa Generator{};
    static initialise = true;
    if (initialise){
        initialise = false;
        try {
            int sherpa_argc = 5;
            char* sherpa_argv[] = {"some_binary","-f","Gun.dat","EXTERNAL_RNG=ProbProbRNG","SHERPA_LDADD=ProbProgRNG"};
            Generator.InitializeTheRun(sherpa_argc,sherpa_argv);
            Generator.InitializeTheEventHandler();
        }
        catch (Exception exception) {
            std::terminate();
        }
    }

    try {
        while(!generator.GenerateOneEvent());
    }
    catch (Exception exception) {
        std::terminate();
    }

    auto ret = jailbreak::instance().m_histo3d;

    Generator.SummarizeRun();
    return ret;
}
