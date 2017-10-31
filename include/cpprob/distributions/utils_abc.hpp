#ifndef INCLUDE_UTILS_ABC
#define INCLUDE_UTILS_ABC

#include "cpprob/distributions/utils_base.hpp"

#include "cpprob/distributions/abc.hpp"

namespace cpprob {

//////////////////////////////
////////// Proposal //////////
//////////////////////////////
template<class Sample, class LogPDF>
struct logpdf<ABC<Sample, LogPDF>> {
    auto operator()(const ABC<Sample, LogPDF> & distr,
                        const typename ABC<Sample, LogPDF>::result_type & x) const
    {
        return distr.logpdf(x);
    }
};

} // end namespace cpprob

#endif //INCLUDE_UTILS_ABC
