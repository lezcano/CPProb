#include "cpprob/sample.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "cpprob/distributions/vmf.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"
#include "cpprob/distributions/distribution_utils.hpp"


namespace cpprob {

////////////////////////////////////////////////////////////////////////////////
////////////////////////            Sample              ////////////////////////
////////////////////////////////////////////////////////////////////////////////

Sample::Sample(const std::string& sample_address,
               const infcomp::protocol::Distribution & proposal_type,
               const boost::any& proposal_distr,
               NDArray<double> value,
               int sample_instance,
               int time_index) :
        sample_address_{sample_address},
        proposal_type_{proposal_type},
        proposal_distr_{proposal_distr},
        value_{value},
        sample_instance_{sample_instance},
        time_index_{time_index} { }

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
        auto mu = distr.mu();
        auto mu_nd = NDArray<double>(mu.begin(), mu.end());
        return infcomp::protocol::CreateVMF(buff,
                                            infcomp::protocol::CreateNDArray(buff,
                                                                             buff.CreateVector<double>(mu_nd.values()),
                                                                             buff.CreateVector<int32_t>(mu_nd.shape())),
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
        auto mean = distr.mean();
        auto sigma = distr.sigma();
        auto mean_nd = NDArray<double>(mean.begin(), mean.end());
        auto sigma_nd = NDArray<double>(sigma.begin(), sigma.end());
        return infcomp::protocol::CreateMultivariateNormal(buff,
                                                           infcomp::protocol::CreateNDArray(buff,
                                                                                            buff.CreateVector<double>(mean_nd.values()),
                                                                                            buff.CreateVector<int32_t>(mean_nd.shape())),
                                                           infcomp::protocol::CreateNDArray(buff,
                                                                                            buff.CreateVector<double>(sigma_nd.values()),
                                                                                            buff.CreateVector<int32_t>(sigma_nd.shape()))
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
