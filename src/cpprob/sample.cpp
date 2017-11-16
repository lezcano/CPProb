#include "cpprob/sample.hpp"

#include "flatbuffers/infcomp_generated.h"  // for Distribution, Sample (ptr...

namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
////////////////////////            Sample              ////////////////////////
////////////////////////////////////////////////////////////////////////////////

flatbuffers::Offset<protocol::Sample> Sample::pack(flatbuffers::FlatBufferBuilder & buff) const
{
    auto serialised_distr = [&] () -> flatbuffers::Offset<void> {
        if (this->distr_enum_ == protocol::Distribution::NONE) {
           return 0;
        }
        else {
            return serialise_distr_(buff, val_);
        }
    }();

    return protocol::CreateSample(
            buff,
            buff.CreateString(addr_),
            distr_enum_,
            serialised_distr);
}

void Sample::set_value(const boost::any &value)
{
    val_ = value;
}

std::string Sample::address() const
{
    return addr_;
}

} // end namespace cpprob
