
#include "sherpa_details.hpp"

#include <string>
#include <vector>

#include "SHERPA/Main/Sherpa.H"
#include "ATOOLS/Org/Exception.H"
#include "ATOOLS/Org/Run_Parameter.H"
#include "ATOOLS/Org/CXXFLAGS.H"
#include "ATOOLS/Org/CXXFLAGS_PACKAGES.H"
#include "ATOOLS/Org/My_MPI.H"
#include "ATOOLS/Org/AnalysisJailbreak.H"

#include <iostream>     // std::cout, std::ostream, std::ios
#include <exception>    // std::terminate



void sherpa_details::sherpa(){
  static ::SHERPA::Sherpa Generator{};
}


std::vector<std::vector<std::vector<double>>> sherpa_details::sherpa_wrapper()
{
    jailbreak::instance().m_histo3d.clear();
    static  ::SHERPA::Sherpa Generator{};
    static bool initialise = true;
    if (initialise){
        initialise = false;
        try {
            int sherpa_argc = 5;
            char* sherpa_argv[] = {"some_binary","-f","Gun.dat","EXTERNAL_RNG=ProbProbRNG","SHERPA_LDADD=ProbProgRNG"};
            Generator.InitializeTheRun(sherpa_argc,sherpa_argv);
            Generator.InitializeTheEventHandler();
        }
        catch (::ATOOLS::Exception exception) {
            std::terminate();
        }
    }

    try {
        while(!Generator.GenerateOneEvent());
    }
    catch (::ATOOLS::Exception exception) {
        std::terminate();
    }

    std::cout << "SHERPAPROBPROG: successfully generated an event!!" << std::endl;
    std::cout << "SHERPAPROBPROG: jailbroken value is: " << jailbreak::instance().m_histo3d.size() << " ... " << std::endl;
    std::cout << "----" << std::endl;

    auto ret = jailbreak::instance().m_histo3d;

    Generator.SummarizeRun();
    return ret;
}
