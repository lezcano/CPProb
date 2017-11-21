#ifndef CPPROB_UTILS_BASE_HPP
#define CPPROB_UTILS_BASE_HPP

namespace cpprob {

template<class Distribution>
struct buffer;

template<class Distr>
using buffer_t = typename buffer<Distr>::type;

template<class Distribution>
struct proposal;

template<class Distr>
using proposal_t = typename proposal<Distr>::type;

template<class Distribution>
struct normalise;

template<class Distribution>
struct from_flatbuffers;

template<class Distribution>
struct to_flatbuffers;

template<class Distribution>
struct logpdf;

}  // end namespace cpprob
#endif //CPPROB_UTILS_BASE_HPP
