#include <cpprob/traits.hpp>
#include "cpprob/trace.hpp"

#include "flatbuffers/infcomp_generated.h"

namespace cpprob{

double Trace::log_w() const{ return log_w_; }

std::vector<std::vector<double>> Trace::x() const{ return x_; }

flatbuffers::Offset<infcomp::Trace> Trace::pack(flatbuffers::FlatBufferBuilder& buff) const{
    std::vector<flatbuffers::Offset<infcomp::Sample>> vec_sample(samples_.size());
    std::transform(samples_.begin(), samples_.end(), vec_sample.begin(),
        [&](const Sample& s){return s.pack(buff);});

    return infcomp::CreateTraceDirect(
            buff,
            infcomp::CreateNDArray(buff,
                buff.CreateVector<double>(observes_),
                buff.CreateVector<int32_t>(std::vector<int32_t>{static_cast<int32_t>(observes_.size())})),
            &vec_sample);
}

Trace& Trace::operator+= (const Trace& rhs){
    if (rhs.x_.size() > this->x_.size())
        this->x_.resize(rhs.x_.size());

    for (std::size_t i = 0; i < rhs.x_.size(); ++i){
        if (rhs.x_[i].empty()) continue;

        if (rhs.x_[i].size() > this->x_[i].size())
            this->x_[i].resize(rhs.x_[i].size());

        // Add the vectors
        std::transform(rhs.x_[i].begin(),
                       rhs.x_[i].end(),
                       this->x_[i].begin(),
                       this->x_[i].begin(),
                       std::plus<double>());
    }
    return *this;
}

Trace& Trace::operator*= (double rhs){
    for (auto& v : this->x_)
        std::transform(v.begin(),
                       v.end(),
                       v.begin(),
                       [rhs](double a){ return rhs * a; });
    return *this;
}

Trace& Trace::operator/= (double rhs){
    for (auto& v : this->x_)
        std::transform(v.begin(),
                       v.end(),
                       v.begin(),
                       [rhs](double a){ return a / rhs; });
    return *this;
}

Trace operator+ (const Trace& lhs, const Trace& rhs){ return Trace(lhs) += rhs; }
Trace operator* (const double lhs, const Trace& rhs){ return Trace(rhs) *= lhs; }
Trace operator* (const Trace& lhs, const double rhs){ return Trace(lhs) *= rhs; }

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[ ";
    for (const auto &elem : v)
        out << elem << " ";
    out << "]";
    return out;
}

std::ostream &operator<<(std::ostream &out, const Trace &v) {
    out << v.x_;
    return out;
}

Sample::Sample(const std::string& sample_address,
           int sample_instance,
           const infcomp::ProposalDistribution& proposal_type,
           const flatbuffers::Offset<void>& proposal,
           int time_index,
           double value) :
        sample_address_{sample_address},
        sample_instance_{sample_instance},
        proposal_type_{proposal_type},
        proposal_{proposal},
        time_index_{time_index},
        value_{value}{}

void Sample::set_value(double value){ value_ = value; }

flatbuffers::Offset<infcomp::Sample> Sample::pack(flatbuffers::FlatBufferBuilder& buff) const{
    return infcomp::CreateSample(
        buff,
        time_index_,
        buff.CreateString(sample_address_),
        sample_instance_,
        proposal_type_,
        proposal_,
        infcomp::CreateNDArray(buff,
            buff.CreateVector<double>(std::vector<double>(value_)),
            buff.CreateVector<int32_t>(std::vector<int32_t>{1})));
}
}
