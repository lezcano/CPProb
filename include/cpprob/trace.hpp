#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>

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

class TraceCompile {
public:

    flatbuffers::Offset<infcomp::Trace> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:

    // Friends
    friend class State;

    // Attributes
    int time_index_ = 1;
    std::vector<int> sample_instance_;
    std::vector<Sample> samples_;
    std::vector<NDArray<double>> observes_;
};

class TracePredicts {
public:
    double log_w() const;

    void add_predict(int id, const NDArray<double>& x);

    void increment_cum_log_prob(double log_p);

    //TracePredicts& operator+=(const TracePredicts &);
    //TracePredicts& operator*=(double);
    //TracePredicts& operator/=(double);

    //friend TracePredicts operator+(const TracePredicts &, const TracePredicts &);
    //friend TracePredicts operator*(double, const TracePredicts &);
    //friend TracePredicts operator*(const TracePredicts &, double);


    template<class CharT, class Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const TracePredicts & t)
    {
        using namespace detail; // operator << for containers
        return os << std::make_pair(t.predict_, t.log_w_);
    }

private:
    // Attributes
    std::vector<std::pair<int, NDArray<double>>> predict_;
    double log_w_ = 0;
};

}  // namespace cpprob
#endif  // INCLUDE_TRACE_HPP_
