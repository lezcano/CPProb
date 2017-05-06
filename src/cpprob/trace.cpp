#include "cpprob/trace.hpp"

#include <exception>
#include <string>
#include <boost/any.hpp>

#include "cpprob/distributions/vmf.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"

#include "cpprob/traits.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob{

double Trace::log_w() const{ return log_w_; }

std::vector<std::vector<NDArray<double>>> Trace::predict() const{ return predict_; }

flatbuffers::Offset<infcomp::Trace> Trace::pack(flatbuffers::FlatBufferBuilder& buff) const{
    std::vector<flatbuffers::Offset<infcomp::Sample>> vec_sample(samples_.size());
    std::transform(samples_.begin(), samples_.end(), vec_sample.begin(),
        [&](const Sample& s){return s.pack(buff);});

    // TODO(Lezcano) We are currently just packing the first element of observes_!!!
    return infcomp::CreateTraceDirect(
            buff,
            infcomp::CreateNDArray(buff,
                buff.CreateVector<double>(observes_[0].values()),
                buff.CreateVector<int32_t>(observes_[0].shape())),
            &vec_sample);
}

Trace& Trace::operator+= (const Trace& rhs){
    if (rhs.predict_.size() > this->predict_.size())
        this->predict_.resize(rhs.predict_.size());

    for (std::size_t i = 0; i < rhs.predict_.size(); ++i){
        if (rhs.predict_[i].empty()) continue;

        if (rhs.predict_[i].size() > this->predict_[i].size())
            this->predict_[i].resize(rhs.predict_[i].size());

        // Add the vectors
        std::transform(rhs.predict_[i].begin(),
                       rhs.predict_[i].end(),
                       this->predict_[i].begin(),
                       this->predict_[i].begin(),
                       std::plus<NDArray<double>>());
    }
    return *this;
}

Trace& Trace::operator*= (double rhs){
    for (auto& v : this->predict_)
        for (auto& e : v)
            e *= rhs;
    return *this;
}

Trace& Trace::operator/= (double rhs){
    for (auto& v : this->predict_)
        for (auto& e : v)
            e /= rhs;
    return *this;
}

Trace operator+ (const Trace& lhs, const Trace& rhs){ return Trace(lhs) += rhs; }
Trace operator* (const double lhs, const Trace& rhs){ return Trace(rhs) *= lhs; }
Trace operator* (const Trace& lhs, const double rhs){ return Trace(lhs) *= rhs; }

Sample::Sample(const std::string& sample_address,
           int sample_instance,
           const infcomp::Distribution & proposal_type,
           const boost::any& proposal_distr,
           int time_index,
           NDArray<double> value) :
        sample_address_{sample_address},
        sample_instance_{sample_instance},
        proposal_type_{proposal_type},
        proposal_distr_{proposal_distr},
        time_index_{time_index},
        value_{value} { }

void Sample::set_value(const NDArray<double>& value){ value_ = value; }

flatbuffers::Offset<infcomp::Sample> Sample::pack(flatbuffers::FlatBufferBuilder& buff) const{
    return infcomp::CreateSample(
        buff,
        time_index_,
        buff.CreateString(sample_address_),
        sample_instance_,
        proposal_type_,
        pack_distr(buff, proposal_distr_, proposal_type_),
        infcomp::CreateNDArray(buff,
            buff.CreateVector<double>(value_.values()),
            buff.CreateVector<int32_t>(value_.shape())));
}

flatbuffers::Offset<void> Sample::pack_distr(flatbuffers::FlatBufferBuilder& buff,
                                             const boost::any& distr_any,
                                             infcomp::Distribution type) const
{
    if (type == infcomp::Distribution::Normal){
        auto distr = boost::any_cast<boost::random::normal_distribution<>>(distr_any);
        return infcomp::CreateNormal(buff, distr.mean(), distr.sigma()).Union();
    }
    else if (type == infcomp::Distribution::UniformDiscrete){
        auto distr = boost::any_cast<boost::random::uniform_smallint<>>(distr_any);
        return infcomp::CreateUniformDiscrete(buff,distr.a(), distr.b()-distr.a()+1).Union();
    }
    else if (type == infcomp::Distribution::VMF){
        auto distr = boost::any_cast<vmf_distribution<>>(distr_any);
        auto mu = NDArray<double>(distr.mu());
        return infcomp::CreateVMF(buff,
                                  infcomp::CreateNDArray(buff,
                                                         buff.CreateVector<double>(mu.values()),
                                                         buff.CreateVector<int32_t>(mu.shape())),
                                  distr.kappa()).Union();
    }
    else if (type == infcomp::Distribution::Poisson){
        auto distr = boost::any_cast<boost::random::poisson_distribution<>>(distr_any);
        return infcomp::CreatePoisson(buff, distr.mean()).Union();

    }
    else if (type == infcomp::Distribution::UniformContinuous){
        auto distr = boost::any_cast<boost::random::uniform_real_distribution<>>(distr_any);
        return infcomp::CreateUniformContinuous(buff, distr.a(), distr.b()).Union();
    }
    else if (type == infcomp::Distribution::MultivariateNormal){
        auto distr = boost::any_cast<multivariate_normal_distribution<>>(distr_any);
        auto mean = NDArray<double>(distr.mean());
        auto sigma = NDArray<double>(distr.sigma());
        return infcomp::CreateMultivariateNormal(buff,
                                                 infcomp::CreateNDArray(buff,
                                                                        buff.CreateVector<double>(mean.values()),
                                                                        buff.CreateVector<int32_t>(mean.shape())),
                                                 infcomp::CreateNDArray(buff,
                                                                        buff.CreateVector<double>(sigma.values()),
                                                                        buff.CreateVector<int32_t>(sigma.shape()))
        ).Union();
    }
    else if (type == infcomp::Distribution::NONE){
        return 0;
    }
    else{
        throw std::runtime_error("Distribution " +
                           std::to_string(static_cast<std::underlying_type_t<infcomp::Distribution>>(type)) +
                          "not implemented.");
    }
}

} // end namespace cpprob
