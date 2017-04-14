#ifndef INCLUDE_UTILS_HPP
#define INCLUDE_UTILS_HPP

#include <type_traits>                          // std::enable_if_t
#include <random>                               // std::random_device
#include <boost/random/mersenne_twister.hpp>    // boost::random::mt19937
#include <boost/random/random_device.hpp>
#include <array>                                // std:: array
#include <functional>                           // std::ref

namespace cpprob {

// Idea from
// http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
template<class T = boost::random::mt19937, std::size_t N = T::state_size>
std::enable_if_t<N != 0, T> seeded_rng(){
    std::array<typename T::result_type, N> random_data;
    std::random_device rd;
    std::generate(random_data.begin(), random_data.end(), std::ref(rd));
    std::seed_seq seeds(random_data.begin(), random_data.end());
    return T{seeds};
}

std::string get_addr();

boost::random::mt19937& get_rng();
}  // namespace cpprob

#endif //INCLUDE_UTILS_HPP
