#include "models/models.hpp"

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_smallint.hpp>

#include "cpprob/cpprob.hpp"
#include "cpprob/distributions/vmf.hpp"
#include "cpprob/distributions/multivariate_normal.hpp"

namespace models {

////////////////////////////////////////////////////////////////////////////////
//                                  MODIFY ME                                 //
////////////////////////////////////////////////////////////////////////////////
void model(const std::array<double, 16> & obs)
{
    hmm<16>(obs);
}


void all_distr() {
    using cpprob::sample;
    using cpprob::predict;
    using cpprob::observe;

    using boost::random::normal_distribution;
    normal_distribution<> normal{1,2};
    auto normal_val = sample(normal, true);
    predict(normal_val);
    observe(normal, normal_val);

    using boost::random::uniform_smallint;
    uniform_smallint<> discrete {2, 7};
    auto discrete_val = sample(discrete, true);
    predict(discrete_val);
    observe(discrete, discrete_val);

    using boost::random::uniform_real_distribution;
    uniform_real_distribution<> rand_unif {2, 9.5};
    auto rand_unif_val = sample(rand_unif, true);
    predict(rand_unif_val);
    observe(rand_unif, rand_unif_val);

    using boost::random::poisson_distribution;
    poisson_distribution<> poiss(0.8);
    auto poiss_val = sample(poiss, true);
    predict(poiss_val);
    observe(poiss, poiss_val);

    using cpprob::vmf_distribution;
    vmf_distribution<> vmf{{1,2,3}, 3};
    auto vmf_val = sample(vmf, true);
    predict(vmf_val);
    observe(vmf, vmf_val);

    using cpprob::multivariate_normal_distribution;
    multivariate_normal_distribution<> multi{{1,2,3,4}, {2,1,5,3}};
    auto sample_multi = sample(multi, true);
    predict(sample_multi);
    observe(multi, sample_multi);
}

} // end namespace models
