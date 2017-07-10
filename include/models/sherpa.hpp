#ifndef SHERPA_HPP
#define SHERPA_HPP

#include <vector>

namespace SHERPA {
	class Sherpa;
}

namespace models {

    class SherpaWrapper {
    public:
        SherpaWrapper();

        void operator()(const std::vector<std::vector<std::vector<double>>> & observes) const;

        std::tuple<int,
                   std::vector<double>,
                   std::vector<std::vector<std::vector<double>>>>
            sherpa() const;

        ~SherpaWrapper();
    private:
        SHERPA::Sherpa* generator_;
    };
} // end namespace modeles
#endif //SHERPA_HPP
