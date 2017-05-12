#ifndef CPPROB_SHERPA_HPP
#define CPPROB_SHERPA_HPP

#include <vector>
#include <memory>

class SHERPA::Sherpa;
namespace sherpa_detail {

    class SherpaWrapper {
    public:
        SherpaWrapper();

        void operator()(const std::vector<std::vector<std::vector<double>>> & observes);

        std::vector<std::vector<std::vector<double>>> sherpa();
    private:
        std::unique_ptr<SHERPA::Sherpa> generator_;
    };
}
#endif //CPPROB_SHERPA_HPP
