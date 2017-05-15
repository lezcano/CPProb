#ifndef CPPROB_SHERPA_HPP
#define CPPROB_SHERPA_HPP

#include <vector>

namespace SHERPA {
	class Sherpa;
}

namespace sherpa_detail {

    class SherpaWrapper {
    public:
        SherpaWrapper();

        void operator()(const std::vector<std::vector<std::vector<double>>> & observes) const;

        std::vector<std::vector<std::vector<double>>> sherpa();
        ~SherpaWrapper();
    private:
        SHERPA::Sherpa* generator_;
    };
}
#endif //CPPROB_SHERPA_HPP
