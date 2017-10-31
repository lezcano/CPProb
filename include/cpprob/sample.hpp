#ifndef CPPROB_SAMPLE_HPP
#define CPPROB_SAMPLE_HPP

#include <functional>                       // for function
#include <string>                           // for string
#include <boost/any.hpp>                    // for any

#include "cpprob/ndarray.hpp"               // for NDArray
#include "cpprob/distributions/utils_distributions.hpp"
#include "flatbuffers/infcomp_generated.h"  // for Distribution, Sample (ptr...

namespace cpprob {

class Sample {
public:
    Sample() = default;

    template<class Distr>
    Sample(const std::string & addr,
           const Distr & distr,
           const NDArray<double> & val = 0,
           int sample_instance = 0,
           int time_index = 0) :
            addr_(addr),
            serialise_distr_(
                    [distr](flatbuffers::FlatBufferBuilder& buff)
                    { return serialise<Distr>::to_flatbuffers(buff, distr); }),
            distr_enum_(infcomp::protocol::DistributionTraits<cpprob::buffer_t<Distr>>::enum_value),
            val_(val),
            sample_instance_(sample_instance),
            time_index_(time_index) {}

    flatbuffers::Offset<infcomp::protocol::Sample> pack(flatbuffers::FlatBufferBuilder & buff) const;

    void set_value(const NDArray<double> & value);

    std::string address() const;

private:

    std::string addr_;
    // Store a function that, given a buffer, serialises the given distribution
    std::function<flatbuffers::Offset<void>(flatbuffers::FlatBufferBuilder& buff)> serialise_distr_;
    infcomp::protocol::Distribution distr_enum_;
    NDArray<double> val_ = 0;
    int sample_instance_ = 0;
    int time_index_ = 0;
};

} // end namespace cpprob
#endif //CPPROB_SAMPLE_HPP
