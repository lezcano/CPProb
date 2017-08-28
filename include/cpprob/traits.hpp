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

namespace detail {
template<typename T>
auto is_object_callable_impl(int)
-> decltype(
        &T::operator(),
        void(), // Handle evil operator ,
        std::true_type{});

template<typename T>
std::false_type is_object_callable_impl(...);
} // end namespace detail

template<typename T>
using is_object_callable = decltype(detail::is_object_callable_impl<T>(0));

template<class T>
using last_elem = std::tuple_element_t<
        std::tuple_size<T>::value - 1,
        T>;

namespace detail {
template<class F>
constexpr std::size_t num_args_impl(std::false_type)
{
    return boost::function_types::function_arity<F>::value;
}

// The first argument is the this pointer
template<class F>
constexpr std::size_t num_args_impl(std::true_type)
{
    return boost::function_types::function_arity<decltype(&F::operator())>::value - 1;
}
} // end namespace detail

template<class F>
constexpr std::size_t num_args()
{
    return detail::num_args_impl<F>(is_object_callable<F>{});
}

template<class F>
auto function_type(std::true_type) -> decltype(&F::operator()) {}
template<class F>
auto function_type(std::false_type) -> F {}

template<class F, template<class ...> class C, class = std::make_index_sequence<num_args<F>()>>
struct parameter_types;

template<class F, template<class ...> class C, std::size_t... Indices>
struct parameter_types<F, C, std::index_sequence<Indices...>> {
    using type = C<std::decay_t<typename boost::mpl::at_c<boost::function_types::parameter_types<
        decltype(function_type<F>(is_object_callable<F>{}))>,
        std::conditional_t<is_object_callable<F>::value,
                           std::integral_constant<int, Indices + 1>,
                           std::integral_constant<int, Indices>>::value
            >::type> ...>;
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
