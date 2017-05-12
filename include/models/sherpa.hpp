#ifndef CPPROB_SHERPA_HPP
#define CPPROB_SHERPA_HPP

#include <vector>

#include "SHERPA/Main/Sherpa.H"

namespace sherpa_detail {

    class SherpaWrapper {
    public:
        SherpaWrapper();

        void operator()(const std::vector<std::vector<std::vector<double>>> & observes);

        std::vector<std::vector<std::vector<double>>> sherpa();
    private:
        SHERPA::Sherpa generator_;
    };
}
#endif //CPPROB_SHERPA_HPP
