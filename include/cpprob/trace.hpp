#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpprob/any.hpp"
#include "cpprob/ndarray.hpp"
#include "cpprob/sample.hpp"

#include "flatbuffers/infcomp_generated.h"

namespace cpprob{
class TraceCompile {
public:

    flatbuffers::Offset<infcomp::protocol::Trace> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:
    // Friends
    friend class StateCompile;

    // Attributes
    int time_index_ = 1;
    std::unordered_map<std::string, int> sample_instance_;
    std::vector<Sample> samples_;
    std::vector<NDArray<double>> observes_;
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
};

}  // namespace cpprob
#endif  // INCLUDE_TRACE_HPP_
