#ifndef INCLUDE_VMF_HPP_
#define INCLUDE_VMF_HPP_

#include <vector>
#include <utility>
#include <iosfwd>
#include <istream>
#include <algorithm>
#include <cmath>

#include <boost/range.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_on_sphere.hpp>

#include "cpprob/detail/vector_io.hpp"

namespace cpprob {

namespace detail {
template<class RealType = double>
void normalize_l2(std::vector <RealType> &vector) {
    auto norm = std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), static_cast<RealType>(0)));
    std::transform(vector.begin(), vector.end(), vector.begin(), [norm](const RealType &a) { return a / norm; });
}

} // end namespace detail

template<class RealType = double>
class vmf_distribution {
public:
    // types
    using input_type = RealType;
    using result_type = std::vector<RealType>;

    // member classes/structs/unions

    class param_type {
    public:
        // types
        using distribution_type = vmf_distribution;

        // construct/copy/destruct
        param_type()
        {
            init_empty();
        }

        template<class Iter>
        param_type(Iter mu_first, Iter mu_last, const RealType& kappa) : mu_(mu_first, mu_last), kappa_{kappa}
        {
            init(mu_first, mu_last, kappa);
        }

        param_type(const std::initializer_list<RealType> & mu, const RealType& kappa) : mu_(mu), kappa_(kappa)
        {
            init(mu.begin(), mu.end(), kappa);
        }

        template<class Range>
        explicit param_type(const Range& range_mu, const RealType& kappa)
        {
            init(boost::begin(range_mu), boost::end(range_mu), kappa);
        }

        // public member functions
        std::vector<RealType> mu() const
        {
            return mu_;
        }

        RealType kappa() const
        {
            return kappa_;
        }

        // friend functions
        template<class CharT, class Traits>
        friend std::basic_ostream< CharT, Traits > &
        operator<<(std::basic_ostream< CharT, Traits > & os, const param_type & param)
        {
            detail::print_vector(os, param.mu_);
            os << os.widen(' ') << param.kappa_;
            return os;
        }

        template<class CharT, class Traits>
        friend std::basic_istream< CharT, Traits > &
        operator>>(std::basic_istream< CharT, Traits > & is, const param_type & param)
        {
            std::vector<RealType> mu_temp;
            detail::read_vector(is, mu_temp);
            if(!is) {
                return is;
            }
            param.mu_.swap(mu_temp);
            is >> std::ws >> param.kappa_;
            return is;
        }

        friend bool operator==(const param_type & lhs, const param_type & rhs)
        {
            return lhs.kappa_ == rhs.kappa_ && lhs.mu_ == rhs.mu_;
        }
        friend bool operator!=(const param_type & lhs, const param_type & rhs)
        {
            return !(lhs == rhs);
        }
    private:

        void init_empty()
        {
            mu_ = {1, 0, 0};
            kappa_ = 1;
        }

        template<class Iter>
        void init(Iter mu_first, Iter mu_last, RealType kappa)
        {
            if (mu_first == mu_last){
                init_empty();
            }
            else{
                mu_ = std::vector<RealType>(mu_first, mu_last);
                kappa_ = kappa;
            }
        }

        friend class vmf_distribution;
        std::vector<RealType> mu_;
        RealType kappa_;
    };

    // construct/copy/destruct
    template<class Iter>
    vmf_distribution(Iter mu_first, Iter mu_last, const RealType& kappa) : param_(mu_first, mu_last, kappa) {}

    vmf_distribution(const std::initializer_list<RealType> & mu, const RealType& kappa) : param_(mu, kappa) {}

    template<class Range>
    explicit vmf_distribution(const Range& range_mu, const RealType& kappa) : param_(range_mu, kappa) {}

    // public member functions
    RealType kappa() const
    {
        return param_.kappa_;
    }

    std::vector<RealType> mu() const
    {
        return param_.mu_;
    }

    param_type param() const
    {
        return param_;
    }

    void param(const param_type & param)
    {
        param_ = param;
    }

    void reset() {}

    template<class URNG> result_type operator()(URNG & rng) const
    {
        auto dim = param_.mu_.size();
        result_type ret(dim);
        auto w = sample_weight(rng, param_.kappa_, dim);
        auto v = sample_orthonormal_to(rng, param_.mu_);
        std::transform(param_.mu_.begin(), param_.mu_.end(), v.begin(), ret.begin(),
                       [w, sqw=std::sqrt(1 - w * w)](RealType mui, RealType vi){ return vi*sqw + w*mui; });
        return ret;
    }

    template<class URNG> result_type operator()(URNG & rng, const param_type & param) const
    {
        return vmf_distribution(param)(rng);
    }

    // friend functions
    template<class CharT, class Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > &os,
               const vmf_distribution & distr)
    {
        os << distr.param_;
        return os;
    }

    template<class CharT, class Traits>
    friend std::basic_istream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is,
               vmf_distribution & distr)
    {
        param_type parm;
        if(is >> parm) {
            distr.param_ = std::move(parm);
        }
        return is;
    }
    friend bool operator==(const vmf_distribution & lhs,
                           const vmf_distribution & rhs)
    {
        return lhs == rhs;
    }
    friend bool operator!=(const vmf_distribution & lhs,
                           const vmf_distribution & rhs)
    {
        return lhs != rhs;
    }
private:

    template<class URNG>
    RealType sample_weight(URNG& urng, const RealType& kappa, int dim) const {
        dim--; // We sample over S^{n-1}
        auto b = dim / (std::sqrt(4.0 * kappa * kappa + dim * dim) + 2 * kappa);
        auto x = (1.0 - b) / (1 + b);
        auto c = kappa * x + dim * std::log(1 - x * x);
        const boost::random::beta_distribution<RealType> beta{dim / static_cast<RealType>(2),
                                                              dim / static_cast<RealType>(2)};
        boost::random::uniform_01<RealType> unif01;

        for (;;) {
            auto z = beta(urng);
            auto w = (1 - (1 + b) * z) / (1 - (1 - b) * z);
            auto u = unif01(urng);
            if (kappa * w + dim * std::log(1 - x * w) - c >= std::log(u))
                return w;
        }
    }

    template<class URNG>
    std::vector<RealType> sample_orthonormal_to(URNG& urng, const std::vector<RealType>& mu) const {
        auto dim = mu.size();
        boost::random::uniform_on_sphere<RealType, std::vector<RealType>> unif_sphere{static_cast<int>(dim)};
        auto v = unif_sphere(urng);

        auto scalar_prod = std::inner_product(v.begin(), v.end(), mu.begin(), static_cast<RealType>(0));

        // Compute a vector orthogonal to mu
        std::vector<RealType> ortho(dim);
        std::transform(v.begin(), v.end(), mu.begin(), ortho.begin(),
                       [scalar_prod](RealType vi, RealType mui){ return vi-mui*scalar_prod; });

        // Normalise orthogonal vector
        auto l2_ortho = std::sqrt(std::inner_product(ortho.begin(), ortho.end(), ortho.begin(), static_cast<RealType>(0)));
        std::transform(ortho.begin(), ortho.end(), ortho.begin(), [l2_ortho](RealType a) { return a / l2_ortho; });
        return ortho;
    }

    param_type param_;
};

} // end namespace cpprob
#endif //INCLUDE_VMF_HPP_
