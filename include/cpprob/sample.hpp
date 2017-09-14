#ifndef CPPROB_SAMPLE_HPP
#define CPPROB_SAMPLE_HPP

#include <string>                           // for string
#include <boost/any.hpp>                    // for any

#include "cpprob/ndarray.hpp"               // for NDArray
#include "flatbuffers/infcomp_generated.h"  // for Distribution, Sample (ptr...

namespace cpprob {

class Sample {
public:
    Sample() = default;

    Sample(const std::string &sample_address,
           const infcomp::protocol::Distribution &proposal_type,
           const boost::any &proposal_distr,
           NDArray<double> value = 0,
           int sample_instance = 0,
           int time_index = 0);

    void set_value(const NDArray<double> &value);

    std::string sample_address();

    flatbuffers::Offset <infcomp::protocol::Sample> pack(flatbuffers::FlatBufferBuilder &buff) const;

private:

    flatbuffers::Offset<void> pack_distr(flatbuffers::FlatBufferBuilder &buff,
                                         const boost::any &distr,
                                         infcomp::protocol::Distribution type) const;

    std::string sample_address_;
    infcomp::protocol::Distribution proposal_type_;
    boost::any proposal_distr_;
    NDArray<double> value_{0};
    int sample_instance_{0};
    int time_index_{0};
};

} // end namespace cpprob
#endif //CPPROB_SAMPLE_HPP
