#ifndef CPPROB_CALL_FUNCTION_HPP
#define CPPROB_CALL_FUNCTION_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <boost/function_types/result_type.hpp>

#include "cpprob/metapriors.hpp"
#include "cpprob/traits.hpp"

namespace cpprob {
namespace detail {
using boost::function_types::result_type;


// Call f default
template<class B, std::size_t I>
auto init_val(std::true_type)
{
    return rem_prior(std::tuple_element_t<I, typename B::types>{});
}

template<class B, std::size_t I>
auto init_val(std::false_type)
{
    return B{};
}

template<class F, std::size_t... Indices>
auto call_f_default_params_detail(const F &f, std::index_sequence<Indices...>, std::true_type)
{
    using B = last_elem<parameter_types_t<F>>;
    return f(init_val<B, Indices>(
            std::integral_constant<bool, Indices < std::tuple_size<parameter_types_t<F>>::value - 1>{}) ...);
}

template<class F, std::size_t... Indices>
auto call_f_default_params_detail(const F &f, std::index_sequence<Indices...>, std::false_type)
{
    return f(std::tuple_element_t<Indices, parameter_types_t<F>>{} ...);
}

// Call f Tuple
template<class F, std::size_t I, class... Args>
auto param(const std::tuple<Args...> & args, std::true_type) {
    return std::get<I>(args);
}

template<class F, std::size_t I, class... Args>
auto param(const std::tuple<Args...> &, std::false_type) {
    return last_elem<parameter_types_t<F>>{};
}

template<class F, class... Args, std::size_t... Indices>
auto call_f_tuple_detail(const F &f, const std::tuple<Args...> &args, std::index_sequence<Indices...>, std::true_type)
{
    return f(param<F, Indices>(args, std::integral_constant<bool, Indices < num_args<F>() - 1>{}) ...);
}
template<class F, class... Args, std::size_t... Indices>
auto call_f_tuple_detail(const F &f, const std::tuple<Args...> &args, std::index_sequence<Indices...>, std::false_type)
{
    return f(std::get<Indices>(args)...);
}
} // end namespace detail

template<class F>
auto call_f_default_params(const F &f)
{
    return detail::call_f_default_params_detail(f, std::make_index_sequence<num_args<F>()>(),
                                                std::integral_constant<bool, has_builder<F>::value>{});
}

template<class F, class... Args>
auto call_f_tuple(const F &f, const std::tuple<Args...> &args)
{
    return detail::call_f_tuple_detail(f, args, std::make_index_sequence<num_args<F>()>(),
                                    std::integral_constant<bool, has_builder<F>::value>{});
}

} // end namespace cpprob

#endif //CPPROB_CALL_FUNCTION_HPP
