#include "cpprob/trace.hpp"

#include <algorithm>                        // for transform
#include <cstdint>                          // for int32_t
#include <exception>                        // for terminate
#include <iostream>                         // for operator<<, endl, basic_o...
#include <memory>                           // for allocator_traits<>::value...
#include <string>                           // for string
#include <unordered_map>                    // for unordered_map, _Node_iter...

#include "cpprob/sample.hpp"                // for Sample
#include "flatbuffers/infcomp_generated.h"  // for CreateNDArray, CreateTrac...

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

flatbuffers::Offset<protocol::Trace> TraceCompile::pack(flatbuffers::FlatBufferBuilder &buff) const {
    std::vector<flatbuffers::Offset<protocol::Sample>> vec_sample(samples_.size());
    std::transform(samples_.begin(), samples_.end(), vec_sample.begin(),
                   [&](const Sample & s) { return s.pack(buff); });

    // TODO(Lezcano) Currently we only support one multidimensional observe or many one-dimensional observes
    if (observes_.size() == 1) {
        return protocol::CreateTraceDirect(
                buff,
                protocol::CreateNDArray(buff,
                    buff.CreateVector<double>(observes_.front().values()),
                    buff.CreateVector<int32_t>(observes_.front().shape())),
                &vec_sample);
    }
    else {
        std::vector<double> obs_flat;
        for (auto obs : observes_) {
            if (!obs.is_scalar()) {
                throw std::runtime_error("Multiple observes where one of them is multidimensional is not supported.\n");
            }
            obs_flat.emplace_back(static_cast<double>(obs));
        }
        return protocol::CreateTraceDirect(
                buff,
                protocol::CreateNDArray(buff,
                    buff.CreateVector<double>(obs_flat),
                    buff.CreateVector<int32_t>(std::vector<int32_t>{static_cast<int32_t>(obs_flat.size())})),
                &vec_sample);
    }
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////         Inference             ////////////////////////
////////////////////////////////////////////////////////////////////////////////
std::unordered_map<std::string, int> TraceInfer::ids_predict_;

int TraceInfer::register_addr_predict(const std::string& addr)
{
    return ids_predict_.emplace(addr, static_cast<int>(ids_predict_.size())).first->second;
}


} // end namespace cpprob
