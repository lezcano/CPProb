#include "cpprob/trace.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>

#include <boost/any.hpp>

#include "cpprob/sample.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
/////////////////////////        Compilation            ////////////////////////
////////////////////////////////////////////////////////////////////////////////

flatbuffers::Offset<infcomp::protocol::Trace> TraceCompile::pack(flatbuffers::FlatBufferBuilder &buff) const {
    std::vector<flatbuffers::Offset<infcomp::protocol::Sample>> vec_sample(samples_.size());
    std::transform(samples_.begin(), samples_.end(), vec_sample.begin(),
                   [&](const Sample &s) { return s.pack(buff); });

    // TODO(Lezcano) Currently we only support one multidimensional observe or many one-dimensional observes
    if (observes_.size() == 1) {
        return infcomp::protocol::CreateTraceDirect(
                buff,
                infcomp::protocol::CreateNDArray(buff,
                                                 buff.CreateVector<double>(observes_.front().values()),
                                                 buff.CreateVector<int32_t>(observes_.front().shape())),
                &vec_sample);
    }
    else {
        std::vector<double> obs_flat;
        for (auto obs : observes_) {
            if (!obs.is_scalar()) {
                std::cerr << "Multiple observes where one of them is multidimensional is not supported.\n" << std::endl;
                std::terminate();
            }
            obs_flat.emplace_back(obs.values().front());
        }
        return infcomp::protocol::CreateTraceDirect(
                buff,
                infcomp::protocol::CreateNDArray(buff,
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
