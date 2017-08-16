#ifndef INCLUDE_TRAITS_HPP_
#define INCLUDE_TRAITS_HPP_

#include <iterator>
#include <type_traits>
#include <tuple>
#include <utility>

#include <boost/mpl/at.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/function_arity.hpp>
#include <boost/function_types/result_type.hpp>

namespace cpprob {

template<class T>
using last_elem = std::tuple_element_t<
        std::tuple_size<T>::value - 1,
        T>;

template<class F, std::enable_if_t<std::is_function<typename std::remove_pointer_t<F>>::value, int> = 0>
constexpr std::size_t num_args()
{
    return boost::function_types::function_arity<F>::value;
}

// The first argument is the this pointer
template<class F, std::enable_if_t<std::is_same<std::void_t<decltype(&F::operator())>, void>::value, int> = 0>
constexpr std::size_t num_args()
{
    return boost::function_types::function_arity<decltype(&F::operator())>::value - 1;
}

template<class F, template<class ...> class C, class = void, class = std::make_index_sequence<num_args<F>()>>
struct parameter_types;

template<class F, template<class ...> class C, std::size_t... Indices>
struct parameter_types<F, C,
        std::enable_if_t<std::is_function<typename std::remove_pointer_t<F>>::value>,
        std::index_sequence<Indices...>> {
    using type = C<std::decay_t<typename boost::mpl::at_c<boost::function_types::parameter_types<F>, Indices>::type> ...>;
};

template<class F, template<class ...> class C, std::size_t... Indices>
struct parameter_types<F, C,
        std::enable_if_t<std::is_same<std::void_t<decltype(&F::operator())>, void>::value>,
        std::index_sequence<Indices...>> {
    using type = C<std::decay_t<typename boost::mpl::at_c<boost::function_types::parameter_types<decltype(&F::operator())>, Indices+1>::type> ...>;
};

template<class F, template<class ...> class C = std::tuple>
using parameter_types_t = typename parameter_types<F, C>::type;

namespace detail {
// Very nice and clean implementation from
// https://stackoverflow.com/questions/13830158/check-if-a-variable-is-iterable

// To allow ADL with custom begin/end
using std::begin;
using std::end;

template<typename T>
auto is_iterable_impl(int)
-> decltype(
begin(std::declval<T &>()) != end(std::declval<T &>()), // begin/end and operator !=
        void(), // Handle evil operator ,
        ++std::declval<decltype(begin(std::declval<T &>())) & >(), // operator ++
        void(*begin(std::declval<T &>())), // operator*
        std::true_type{});

template<typename T>
std::false_type is_iterable_impl(...);

using std::tuple_size;
using std::get;

template<typename T>
auto is_tuple_like_impl(int)
-> decltype(
        get<0>(std::declval<T>()),
        void(tuple_size<T>::value),
        std::true_type{});

template<typename T>
std::false_type is_tuple_like_impl(...);
} // end namespace detail

template<typename T>
using is_iterable = decltype(detail::is_iterable_impl<T>(0));
template<typename T>
using is_tuple_like = decltype(detail::is_tuple_like_impl<T>(0));




} // end namespace cpprob
#endif  // INCLUDE_TRAITS_HPP_
