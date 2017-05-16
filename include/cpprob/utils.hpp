#ifndef INCLUDE_UTILS_HPP
#define INCLUDE_UTILS_HPP

#include <type_traits>                          // std::enable_if_t
#include <random>                               // std::random_device
#include <array>                                // std:: array
#include <functional>                           // std::ref
#include <string>
#include <tuple>
#include <sstream>
#include <fstream>
#include <type_traits>

#include <boost/random/mersenne_twister.hpp>    // boost::random::mt19937
#include <boost/random/random_device.hpp>

#include <boost/type_traits/has_less.hpp>

#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/function_arity.hpp>


namespace cpprob {
namespace detail {

template<class Iter, class = std::enable_if_t<
        boost::has_less<
                typename std::iterator_traits<Iter>::value_type>::value>>
typename std::iterator_traits<Iter>::value_type supremum (Iter begin, Iter end)
{
    return *std::max_element(begin, end);
}

template<class T, class = std::enable_if_t< std::is_arithmetic<T>::value> >
T get_zero (T)
{
    return T(0);
}

using boost::function_types::result_type;

// TODO(Lezcano) Use cpprob::parameter_types_t to avoid duplicate code
template<class F, size_t... Indices>
typename result_type<F>::type
call_f_default_params_detail(const F& f, std::index_sequence<Indices...>)
{
    return f(std::decay_t<typename boost::mpl::at_c<boost::function_types::parameter_types<F>, Indices>::type>()...);
}

template <class F, class... Args, size_t... Indices>
typename result_type<F>::type
call_f_tuple_detail(const F& f, const std::tuple<Args...>& args, std::index_sequence<Indices...>)
{
    return f(std::get<Indices>(args)...);
}


// Idea from
// http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
template<class T = boost::random::mt19937, std::size_t N = T::state_size>
std::enable_if_t<N != 0, T> seeded_rng()
{
    std::array<typename T::result_type, N> random_data;
    std::random_device rd;
    std::generate(random_data.begin(), random_data.end(), std::ref(rd));
    std::seed_seq seeds(random_data.begin(), random_data.end());
    return T{seeds};
}

} // end namespace detail

template <class F>
typename boost::function_types::result_type<F>::type
call_f_default_params(const F& f)
{
    return detail::call_f_default_params_detail(f, std::make_index_sequence<boost::function_types::function_arity<F>::value>());
}


template <class F, class... Args>
typename boost::function_types::result_type<F>::type
call_f_tuple(const F& f, const std::tuple<Args...>& args)
{
    static_assert(sizeof...(Args) == boost::function_types::function_arity<F>::value,
                  "Number of arguments and arity of function do not agree.");
    return detail::call_f_tuple_detail(f, args, std::make_index_sequence<boost::function_types::function_arity<F>::value>());
}


std::string get_addr();

boost::random::mt19937& get_rng();

}  // namespace cpprob

#endif //INCLUDE_UTILS_HPP
