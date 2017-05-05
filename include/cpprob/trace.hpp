#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <vector>
#include <iostream>
#include <string>

#include <boost/any.hpp>

#include "cpprob/trace.hpp"
#include "cpprob/ndarray.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob{

class Sample{
public:
    Sample() = default;

    Sample(const std::string& sample_address,
           int sample_instance,
           const infcomp::Distribution& proposal_type,
           const boost::any& proposal_distr,
           int time_index = 0,
           NDArray<double> value = 0);

    void set_value(const NDArray<double>& value);

    flatbuffers::Offset<infcomp::Sample> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:

    flatbuffers::Offset<void> pack_distr(flatbuffers::FlatBufferBuilder& buff,
                                         const boost::any& distr,
                                         infcomp::Distribution type) const;

    std::string sample_address_{};
    int sample_instance_{0};
    infcomp::Distribution proposal_type_;
    boost::any proposal_distr_;
    int time_index_{0};
    NDArray<double> value_{0};
};

class Trace {
public:
    double log_w() const;

    std::vector<std::vector<NDArray<double>>> predict() const;

    flatbuffers::Offset<infcomp::Trace> pack(flatbuffers::FlatBufferBuilder& buff) const;

    Trace& operator+=(const Trace &);
    Trace& operator*=(double);
    Trace& operator/=(double);

    friend Trace operator+(const Trace &, const Trace &);
    friend Trace operator*(double, const Trace &);
    friend Trace operator*(const Trace &, double);

    template<class CharT, class Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const Trace & t)
    {
        detail::print_vector(os, t.predict_);
        return os;
    }

private:

    friend class State;

    int time_index_ = 1;
    double log_w_ = 0;

    std::vector<std::vector<NDArray<double>>> predict_;
    std::vector<std::pair<int, NDArray<double>>> predict_addr_;

    std::vector<int> sample_instance_;

    std::vector<Sample> samples_;
    std::vector<NDArray<double>> observes_;
};

}  // namespace cpprob
#endif  // INCLUDE_TRACE_HPP_
