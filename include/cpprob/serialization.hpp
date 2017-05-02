#ifndef INCLUDE_SERIALIZATION_HPP
#define INCLUDE_SERIALIZATION_HPP

#include <sstream>
#include <tuple>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include "cpprob/detail/vector_io.hpp"

#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/function_arity.hpp>


namespace cpprob {
namespace detail {

template<class CharT, class Traits, class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
static bool load(std::basic_istream<CharT, Traits> &is, T& v)
{
    is >> std::ws >> v;
    return static_cast<bool>(is);
}

// Forward declaration
template<class CharT, class Traits, class... T>
bool load(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup);

template<class CharT, class Traits, class U, class V>
bool load(std::basic_istream<CharT, Traits> &is, std::pair<U, V>& pair)
{
    std::tuple<U, V> tup;
    bool ret = load(is, tup);
    pair.first = std::move(std::get<0>(tup));
    pair.second = std::move(std::get<1>(tup));
    return ret;

}

template<class CharT, class Traits, class... T, size_t... Indices>
bool load_tuple_impl(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup, std::index_sequence<Indices...>)
{
    bool result = true;
    (void)std::initializer_list<int>{ (result = result && load(is, std::get<Indices>(tup)), 0)... };
    return result;
}

template<class CharT, class Traits, class... T>
bool load(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup) {
    CharT ch;
    if (!(is >> ch)) {
        return false;
    }
    if (ch != is.widen('(')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return false;
    }

    load_tuple_impl<CharT, Traits, T...>(is, tup, std::make_index_sequence<sizeof...(T)>());

    if (is.fail()) {
        return false;
    }

    if (!(is >> ch)) {
        return false;
    }
    if (ch != is.widen(')')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }

    return static_cast<bool>(is);
}

template<class CharT, class Traits, class T>
bool load(std::basic_istream<CharT, Traits> &is, std::vector<T>& vec)
{
    CharT ch;
    if (!(is >> ch)) {
        return false;
    }
    if (ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return false;
    }
    do {
        T val;
        if(!load(is, val))
            break;
        vec.push_back(val);
    } while (true);

    if (is.fail()) {
        return false;
    }

    if (!(is >> ch)) {
        return false;
    }
    if (ch != is.widen(']')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
    return static_cast<bool>(is);
}

template<class... T, class CharT, class Traits, size_t... Indices>
void parse_param_impl(std::basic_istream<CharT, Traits>& is, std::tuple<T...>& tup, std::index_sequence<Indices ...>)
{
    (void)std::initializer_list<int>{ (load(is, std::get<Indices>(tup)), 0)... };
}


template <class T>
T load_csv_impl(boost::archive::text_iarchive& f)
{
    T ret;
    f >> ret;
    return ret;
}
} // end namespace detail


template <class... T>
std::tuple<T...> load_csv(const std::string& path)
{
    std::ifstream file(path);
    boost::archive::text_iarchive boost_in(file);
    return std::make_tuple(detail::load_csv_impl<T>(boost_in) ...);
}

template <class... T>
std::tuple<T...> parse_param(const std::string& param)
{
    std::istringstream iss(param);
    std::tuple<T...> tup;
    detail::parse_param_impl(iss, tup, std::make_index_sequence<sizeof...(T)>());
    return tup;
}

namespace detail {
template <class F, size_t... Indices>
auto load_csv_f_impl(const std::string& path, std::index_sequence<Indices...>)
{
    return load_csv<typename boost::mpl::at_c<boost::function_types::parameter_types<F>, Indices>::type ...>(path);
}

template <class F, size_t... Indices>
auto parse_param_f_impl(const std::string& observes, std::index_sequence<Indices...>)
{
    return parse_param<typename boost::mpl::at_c<boost::function_types::parameter_types<F>, Indices>::type ...>(observes);
}

} // namespace detail

template <class F>
auto load_csv_f(const std::string& path)
{
    return detail::load_csv_f_impl<F>(path, std::make_index_sequence<boost::function_types::function_arity<F>::value>());
}

template <class F>
auto parse_param_f(const std::string& observes)
{
    return detail::parse_param_f_impl<F>(observes, std::make_index_sequence<boost::function_types::function_arity<F>::value>());
}

} // end namespace cpprob
#endif //INCLUDE_SERIALIZATION_HPP
