#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <string>                     // for string
#include <unordered_map>              // for unordered_map
#include <utility>                    // for pair
#include <vector>                     // for vector

#include "cpprob/any.hpp"             // for any
#include "cpprob/ndarray.hpp"         // for NDArray
#include "cpprob/sample.hpp"          // for Sample
#include "flatbuffers/infcomp_generated.h"
#include "flatbuffers/flatbuffers.h"  // for FlatBufferBuilder (ptr only)

namespace protocol { struct Trace; }

namespace cpprob {

class TraceCompile {
public:

    flatbuffers::Offset<protocol::Trace> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:
    // Friends
    friend class StateCompile;

    // Attributes
    std::vector<Sample> samples_;
    std::vector<Sample> samples_rejection_;
    std::vector<cpprob::NDArray<double>> observes_;
};

class TraceInfer {
public:

    static int register_addr_predict(const std::string& addr);

private:
    // Friends
    friend class StateInfer;

    // Static Members
    static std::unordered_map<std::string, int> ids_predict_;


    // Attributes
    // We have to separate them so we can dump them in different files.
    // We still use cpprob::any so we do not lose precision
    std::vector<std::pair<int, cpprob::any>> predict_int_;
    std::vector<std::pair<int, cpprob::any>> predict_real_;
    std::vector<std::pair<int, cpprob::any>> predict_any_;
    double log_w_ = 0;

    Sample prev_sample_;
    Sample curr_sample_;
};

}  // namespace cpprob
#endif  // INCLUDE_TRACE_HPP_
