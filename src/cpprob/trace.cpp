#include "cpprob/trace.hpp"

#include <exception>
#include <string>
#include <boost/any.hpp>

#include "cpprob/distributions/vmf.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"

#include "cpprob/traits.hpp"
#include "flatbuffers/infcomp_generated.h"

namespace cpprob {

flatbuffers::Offset<infcomp::protocol::Trace> TraceCompile::pack(flatbuffers::FlatBufferBuilder &buff) const {
    std::vector<flatbuffers::Offset<infcomp::protocol::Sample>> vec_sample(samples_.size());
    std::transform(samples_.begin(), samples_.end(), vec_sample.begin(),
                   [&](const Sample &s) { return s.pack(buff); });

    // TODO(Lezcano) Currently we only support one multidimensional observe or many one-dimensional observes
    if (observes_.size() == 1) {
    return infcomp::protocol::CreateTraceDirect(
            buff,
            infcomp::protocol::CreateNDArray(buff,
                                             buff.CreateVector<double>(observes_.front().values()),
                                             buff.CreateVector<int32_t>(observes_.front().shape())),
            &vec_sample);
    }
    else {
        std::vector<double> obs_flat;
        for (auto obs : observes_) {
            obs_flat.emplace_back(obs.values().front());
        }
        return infcomp::protocol::CreateTraceDirect(
                buff,
                infcomp::protocol::CreateNDArray(buff,
                                                 buff.CreateVector<double>(obs_flat),
                                                 buff.CreateVector<int32_t>(std::vector<int32_t>{static_cast<int32_t>(obs_flat.size())})),
                &vec_sample);
    }
}

double TracePredicts::log_w() const
{
    return log_w_;
}

void TracePredicts::add_predict(int id, const NDArray<double> &x)
{
    predict_.emplace_back(id, x);
}

void TracePredicts::increment_cum_log_prob(double log_p)
{
    log_w_ += log_p;
}

Sample::Sample(const std::string& sample_address,
           int sample_instance,
           const infcomp::protocol::Distribution & proposal_type,
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

flatbuffers::Offset<infcomp::protocol::Sample> Sample::pack(flatbuffers::FlatBufferBuilder& buff) const{
    return infcomp::protocol::CreateSample(
        buff,
        time_index_,
        buff.CreateString(sample_address_),
        sample_instance_,
        proposal_type_,
        pack_distr(buff, proposal_distr_, proposal_type_),
        infcomp::protocol::CreateNDArray(buff,
            buff.CreateVector<double>(value_.values()),
            buff.CreateVector<int32_t>(value_.shape())));
}

flatbuffers::Offset<void> Sample::pack_distr(flatbuffers::FlatBufferBuilder& buff,
                                             const boost::any& distr_any,
                                             infcomp::protocol::Distribution type) const
{
    if (type == infcomp::protocol::Distribution::Normal){
        auto distr = boost::any_cast<boost::random::normal_distribution<>>(distr_any);
        return infcomp::protocol::CreateNormal(buff, distr.mean(), distr.sigma()).Union();
    }
    else if (type == infcomp::protocol::Distribution::UniformDiscrete){
        auto distr = boost::any_cast<boost::random::uniform_smallint<>>(distr_any);
        return infcomp::protocol::CreateUniformDiscrete(buff,distr.a(), distr.b()-distr.a()+1).Union();
    }
    else if (type == infcomp::protocol::Distribution::Discrete){
        auto distr = boost::any_cast<boost::random::discrete_distribution<>>(distr_any);
        // distr.max() + 1 is the number of parameters of the distribution
        return infcomp::protocol::CreateDiscrete(buff, distr.max() + 1).Union();
    }
    else if (type == infcomp::protocol::Distribution::VMF){
        auto distr = boost::any_cast<vmf_distribution<>>(distr_any);
        auto mu = NDArray<double>(distr.mu());
        return infcomp::protocol::CreateVMF(buff,
                                  infcomp::protocol::CreateNDArray(buff,
                                                         buff.CreateVector<double>(mu.values()),
                                                         buff.CreateVector<int32_t>(mu.shape())),
                                  distr.kappa()).Union();
    }
    else if (type == infcomp::protocol::Distribution::Poisson){
        auto distr = boost::any_cast<boost::random::poisson_distribution<>>(distr_any);
        return infcomp::protocol::CreatePoisson(buff, distr.mean()).Union();

    }
    else if (type == infcomp::protocol::Distribution::UniformContinuous){
        auto distr = boost::any_cast<boost::random::uniform_real_distribution<>>(distr_any);
        return infcomp::protocol::CreateUniformContinuous(buff, distr.a(), distr.b()).Union();
    }
    else if (type == infcomp::protocol::Distribution::MultivariateNormal){
        auto distr = boost::any_cast<multivariate_normal_distribution<>>(distr_any);
        auto mean = NDArray<double>(distr.mean());
        auto sigma = NDArray<double>(distr.sigma());
        return infcomp::protocol::CreateMultivariateNormal(buff,
                                                 infcomp::protocol::CreateNDArray(buff,
                                                                        buff.CreateVector<double>(mean.values()),
                                                                        buff.CreateVector<int32_t>(mean.shape())),
                                                 infcomp::protocol::CreateNDArray(buff,
                                                                        buff.CreateVector<double>(sigma.values()),
                                                                        buff.CreateVector<int32_t>(sigma.shape()))
        ).Union();
    }
    else if (type == infcomp::protocol::Distribution::NONE){
        return 0;
    }
    else{
        throw std::runtime_error("Distribution " +
                           std::to_string(static_cast<std::underlying_type_t<infcomp::protocol::Distribution>>(type)) +
                          "not implemented.");
    }
}

} // end namespace cpprob
