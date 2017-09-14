#ifndef INCLUDE_UTILS_HPP
#define INCLUDE_UTILS_HPP

#include <algorithm>                            // std::max_element
#include <array>                                // std:: array
#include <fstream>
#include <functional>                           // std::ref
#include <random>                               // std::random_device
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>                          // std::enable_if_t

#include <boost/type_traits/has_less.hpp>

namespace cpprob {
namespace detail {

template<class Iter, std::enable_if_t< boost::has_less< typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
typename std::iterator_traits<Iter>::value_type supremum (Iter begin, Iter end)
{
    return *std::max_element(begin, end);
}

template<class T, std::enable_if_t< std::is_arithmetic<T>::value, int> = 0>
T get_zero (T)
{
    return T(0);
}



// Idea from
// http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
template<class T = std::mt19937, std::size_t N = T::state_size, std::enable_if_t<N != 0, int> = 0>
T seeded_rng()
{
    std::array<typename T::result_type, N> random_data;
    std::random_device rd;
    std::generate(random_data.begin(), random_data.end(), std::ref(rd));
    std::seed_seq seeds(random_data.begin(), random_data.end());
    return T{seeds};
}

} // end namespace detail


std::string get_addr();

std::mt19937& get_rng();

}  // namespace cpprob

#endif //INCLUDE_UTILS_HPP
