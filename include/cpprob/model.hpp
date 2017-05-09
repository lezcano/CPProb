#ifndef CPPROB_MODEL_HPP
#define CPPROB_MODEL_HPP

#include <pair>
#include <string>
#include <vector>

#include "cpprob/serialization.hpp"
#include "cpprob/ndarray.hpp"

namespace cpprob {

class Model {


    template<class CharT, class Traits>
    void load_points(std::basic_istream< CharT, Traits > & is, const Model & m)
    {
        using namespace detail;

        std::pair< // ([(addr, value)], weight) - Weighted Trace
                std::vector< // [(addr, value)] - Trace
                        std::pair<int, NDArray<double>> // (addr, value)
                >
                , double> val;
        while(is >> val)
            points_.emplace_back(std::move(val));
    }

    template<class CharT, class Traits>
    void load_ids(std::basic_istream< CharT, Traits > & is, const Model & m)
    {
        using namespace detail;
        is >> m.ids_;
    }

private:

    // Attributes
    std::vector< // [([(addr, value)], weight)] - List of Weighted Traces
        std::pair< // ([(addr, value)], weight) - Weighted Trace
            std::vector< // [(addr, value)] - Trace
                std::pair<int, NDArray<double>> // (addr, value)
            >
        , double>
    > points_;
    std::vector<std::string> ids_;

};
}

#endif //CPPROB_MODEL_HPP
