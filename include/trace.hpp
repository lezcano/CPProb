#ifndef INCLUDE_TRACE_HPP_
#define INCLUDE_TRACE_HPP_

#include <vector>
#include <ostream>
#include <string>

#include <msgpack.hpp>

namespace cpprob{
class Sample{
public:
    Sample(int time_index,
           int sample_instance,
           double value,
           const std::string& proposal_name,
           const std::string& sample_address);

    void pack(msgpack::packer<msgpack::sbuffer>& pk) const;

private:
    int time_index_;
    int sample_instance_;
    double value_;
    std::string proposal_name_;
    std::string sample_address_;
};

class Trace {
public:

    double log_w() const;

    std::vector<std::vector<double>> x() const;

    void pack(msgpack::packer<msgpack::sbuffer> &pk);

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

class PrevSampleInference;

struct SampleInference {
    std::string sample_address;
    int sample_instance;
    std::string proposal_name;

    friend class PrevSampleInference;
};

struct PrevSampleInference {
    PrevSampleInference() = default;
    PrevSampleInference &operator=(const PrevSampleInference &) = default;
    PrevSampleInference(const SampleInference &s);
    PrevSampleInference &operator=(const SampleInference &s);

    std::string prev_sample_address = "";
    int prev_sample_instance = 0;
    double prev_sample_value = 0;
};
}
#endif  // INCLUDE_TRACE_HPP_
