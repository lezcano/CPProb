#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <vector>
#include <iostream>
#include <string>

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
           const flatbuffers::Offset<void>& proposal,
           int time_index=0,
           NDArray<double> value=0);

    void set_value(NDArray<double> value);

    flatbuffers::Offset<infcomp::Sample> pack(flatbuffers::FlatBufferBuilder& buff) const;

private:
    std::string sample_address_;
    int sample_instance_;
    infcomp::Distribution proposal_type_;
    flatbuffers::Offset<void> proposal_;
    int time_index_;
    NDArray<double> value_;
};

class Trace {
public:
    double log_w() const;

    std::vector<std::vector<NDArray<double>>> x() const;

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
        detail::print_vector(os, t.x_);
        os << os.widen(' ');
        detail::print_vector(os, t.y_);
        return os;
    }

private:

    template<template <class ...> class Distr, class ...Params>
    friend void observe(Distr<Params ...>& distr, typename Distr<Params ...>::result_type x);

    template<template<class ...> class Distr, class ...Params>
    friend typename Distr<Params ...>::result_type
    sample_impl(Distr<Params ...> &distr, const bool from_observe);

    int time_index_ = 1;
    double log_w_ = 0;
    std::vector<std::vector<NDArray<double>>> x_;

    std::vector<std::pair<NDArray<double>, int>> x_addr_;
    std::vector<double> y_;

    std::vector<Sample> samples_;
    std::vector<NDArray<double>> observes_;
};

}  // namespace cpprob
#endif  // INCLUDE_TRACE_HPP_
