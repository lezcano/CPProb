#include "cpprob/sample.hpp"

#include "flatbuffers/infcomp_generated.h"  // for Distribution, Sample (ptr...

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
////////////////////////            Sample              ////////////////////////
////////////////////////////////////////////////////////////////////////////////

flatbuffers::Offset<infcomp::protocol::Sample> Sample::pack(flatbuffers::FlatBufferBuilder & buff) const
{
    auto serialised_distr = [&] () -> flatbuffers::Offset<void> {
        if (this->distr_enum_ == infcomp::protocol::Distribution::NONE) {
           return 0;
        }
        else {
            return serialise_distr_(buff);
        }
    }();

    return infcomp::protocol::CreateSample(
            buff,
            time_index_,
            buff.CreateString(addr_),
            sample_instance_,
            distr_enum_,
            serialised_distr,
            infcomp::protocol::CreateNDArray(buff,
                                             buff.CreateVector<double>(val_.values()),
                                             buff.CreateVector<int32_t>(val_.shape())));
}

void Sample::set_value(const NDArray<double> &value)
{
    val_ = value;
}

std::string Sample::address() const
{
    return addr_;
}

} // end namespace cpprob
