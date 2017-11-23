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
           const boost::any & val = boost::any{}) :
            addr_(addr),
            serialise_distr_(
                    [distr](flatbuffers::FlatBufferBuilder& buff, const boost::any & value)
                    {
                        using boost::any_cast;
                        if(value.empty()) {
                            return to_flatbuffers<Distr>()(buff, distr, typename Distr::result_type{});
                        }
                        else {
                            return to_flatbuffers<Distr>()(buff, distr, any_cast<typename Distr::result_type>(value));
                        }
                    }),
            val_{val},
            distr_enum_(protocol::DistributionTraits<cpprob::buffer_t<Distr>>::enum_value) {}

    flatbuffers::Offset<protocol::Sample> pack(flatbuffers::FlatBufferBuilder & buff) const;

    void set_value(const boost::any &value);

    std::string address() const;

private:

    std::string addr_;
    // Store a function that, given a buffer, serialises the given distribution
    std::function<flatbuffers::Offset<void>(flatbuffers::FlatBufferBuilder& buff, const boost::any&)> serialise_distr_;
    boost::any val_;
    protocol::Distribution distr_enum_;
};

} // end namespace cpprob
#endif //CPPROB_SAMPLE_HPP
