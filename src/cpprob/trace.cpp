#include <msgpack.hpp>
#include "trace.hpp"

namespace cpprob{

double Trace::log_w() const{ return log_w_; }

std::vector<std::vector<double>> Trace::x() const{ return x_; }

void Trace::pack(msgpack::packer<msgpack::sbuffer>& pk){
    pk.pack_map(2);
        pk.pack(std::string("samples"));
            pk.pack_array(samples_.size());
            for(const auto& s : samples_)
                s.pack(pk);

        pk.pack(std::string("observes"));
        pk.pack_map(2);
            pk.pack(std::string("shape"));
                pk.pack_array(1);
                pk.pack(observes_.size());
            pk.pack(std::string("data"));
            pk.pack(observes_);
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

Sample::Sample(int time_index,
               int sample_instance,
               double value,
               const std::string& proposal_name,
               const std::string& sample_address) :
        time_index_{time_index},
        sample_instance_{sample_instance},
        value_{value},
        proposal_name_{proposal_name},
        sample_address_{sample_address}{}

void Sample::pack(msgpack::packer<msgpack::sbuffer>& pk) const {
    pk.pack_map(5);
    pk.pack(std::string("time-index"));
    pk.pack(time_index_);
    pk.pack(std::string("proposal-name"));
    pk.pack(proposal_name_);
    pk.pack(std::string("value"));
    pk.pack(value_);
    pk.pack(std::string("sample-instance"));
    pk.pack(sample_instance_);
    pk.pack(std::string("sample-address"));
    pk.pack(sample_address_);
}


PrevSampleInference::PrevSampleInference(const SampleInference& s) :
        prev_sample_address{s.sample_address},
        prev_sample_instance{s.sample_instance}{}

PrevSampleInference& PrevSampleInference::operator=(const SampleInference& s){
    prev_sample_address = s.sample_address;
    prev_sample_instance = s.sample_instance;
    return *this;
}

}
