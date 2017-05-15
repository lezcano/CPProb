#ifndef CPPROB_SHERPA_HPP
#define CPPROB_SHERPA_HPP

#include <vector>

namespace SHERPA {
	class Sherpa;
}

namespace cpprob {
namespace models {

    class SherpaWrapper {
    public:
        SherpaWrapper();

        void operator()(const std::vector<std::vector<std::vector<double>>> & observes) const;

        std::vector<std::vector<std::vector<double>>> sherpa() const;


        std::tuple<double,
                std::vector<double>,
                std::vector<std::vector<std::vector<double>>>>
            sherpa_pred_obs() const;

        ~SherpaWrapper();
    private:
        SHERPA::Sherpa* generator_;
    };
} // end namespace modeles
} // end namespace cpprob
#endif //CPPROB_SHERPA_HPP
