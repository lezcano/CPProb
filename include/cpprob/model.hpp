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
    friend std::basic_ostream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is, const Model & m)
    {
        using namespace detail;

        std::pair< // ([(addr, value)], weight) - Weighted Trace
                std::vector< // [(addr, value)] - Trace
                        std::pair<int, NDArray<double>> // (addr, value)
                >
                , double> val;
        while(is >> val)
            points.emplace_back(std::move(val));
        return is;
    }

private:

    // Attributes
    std::vector< // [([(addr, value)], weight)] - List of Weighted Traces
        std::pair< // ([(addr, value)], weight) - Weighted Trace
            std::vector< // [(addr, value)] - Trace
                std::pair<int, NDArray<double>> // (addr, value)
            >
        , double>
    > points;
};
}

#endif //CPPROB_MODEL_HPP
