#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <vector>
#include <ostream>
#include <string>

#include "flatbuffers/infcomp_generated.h"

namespace cpprob{
class Sample{
public:
    Sample() = default;

    Sample(const std::string& sample_address,
           int sample_instance,
           const infcomp::ProposalDistribution& proposal_type,
           const flatbuffers::Offset<void>& proposal,
           int time_index=0,
           double value=0);

    void set_value(double value);

    flatbuffers::Offset<infcomp::Sample> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:
    std::string sample_address_;
    int sample_instance_;
    infcomp::ProposalDistribution proposal_type_;
    flatbuffers::Offset<void> proposal_;
    int time_index_;
    double value_;
};

class Trace {
public:
    double log_w() const;

    std::vector<std::vector<double>> x() const;

    flatbuffers::Offset<infcomp::Trace> pack(flatbuffers::FlatBufferBuilder& buff) const;

    Trace& operator+=(const Trace &);
    Trace& operator*=(double);
    Trace& operator/=(double);

private:

    template<template<class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...> &distr, double x);

    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type
    sample_impl(Distr<Params ...> &distr, const bool from_observe);

    friend std::ostream &operator<<(std::ostream &out, const Trace &v);

    int time_index_ = 1;
    double log_w_ = 0;
    std::vector<std::vector<double>> x_;

    std::vector<std::pair<double, int>> x_addr_;
    std::vector<double> y_;

    std::vector<Sample> samples_;
    std::vector<double> observes_;
};

Trace operator+(const Trace &, const Trace &);
Trace operator*(double, const Trace &);
Trace operator*(const Trace &, double);
}
#endif  // INCLUDE_TRACE_HPP_
