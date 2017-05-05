#ifndef INCLUDE_SERIALIZATION_HPP
#define INCLUDE_SERIALIZATION_HPP

#include <tuple>
#include <vector>
#include <sstream>
#include <fstream>

#include "cpprob/detail/vector_io.hpp"

namespace cpprob {
namespace detail {

// Forward declarations
template<class CharT, class Traits, class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
void load(std::basic_istream<CharT, Traits> &is, T& v);
template<class CharT, class Traits, class U, class V>
void load(std::basic_istream<CharT, Traits> &is, std::pair<U, V>& pair);
template<class CharT, class Traits, class... T>
void load(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup);
template<class CharT, class Traits, class T>
void load(std::basic_istream<CharT, Traits> &is, std::vector<T>& vec);

template<class CharT, class Traits, class T, class = std::enable_if_t<std::is_arithmetic<T>::value>>
void load(std::basic_istream<CharT, Traits> &is, T& v)
{
    is >> std::ws >> v;
}

template<class CharT, class Traits, class U, class V>
void load(std::basic_istream<CharT, Traits> &is, std::pair<U, V>& pair)
{
    std::tuple<U, V> tup;
    load(is, tup);
    if (is.fail())
        return;
    pair.first = std::move(std::get<0>(tup));
    pair.second = std::move(std::get<1>(tup));
}

template<class CharT, class Traits, class... T, size_t... Indices>
void load_tuple_impl(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup, std::index_sequence<Indices...>)
{
    (void)std::initializer_list<int>{ ((load(is, std::get<Indices>(tup)), is.fail()), 0)... };
}

template<class CharT, class Traits, class... T>
void load(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup) {
    CharT ch;
    if (!(is >> ch)) {
        return;
    }
    if (ch != is.widen('(')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return;
    }

    // load_tuple_impl will read exactly the number of elements of the tuple
    // If the is.fail() is set it is because there was an error
    load_tuple_impl<CharT, Traits, T...>(is, tup, std::make_index_sequence<sizeof...(T)>());
    if (is.fail())
        return;

    if (!(is >> ch)) {
        return;
    }
    if (ch != is.widen(')')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
}

template<class CharT, class Traits, class T>
void load(std::basic_istream<CharT, Traits> &is, std::vector<T>& vec)
{
    CharT ch;
    if (!(is >> ch)) {
        return;
    }
    if (ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return;
    }

    do {
        // Needed to be declared inside the loop, otherwise std::move
        // could leave it in an invalid state
        T val;
        load(is, val);
        if(is.fail())
            break;
        vec.emplace_back(std::move(val));
    } while (true);

    // Remark: This accepts not properly specified vectors like
    // [(1 2) (1 4] for std::vector<std::pair<int, int>> or [] for std::vector<std::vector<T>>
    // but well...
    is.clear();
    if (!(is >> ch)) {
        return;
    }
    if (ch != is.widen(']')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
}

template<class... T, class CharT, class Traits, size_t... Indices>
void parse_stream(std::basic_istream<CharT, Traits>& is, std::tuple<T...>& tup, std::index_sequence<Indices ...>)
{
    (void)std::initializer_list<int>{ (load(is, std::get<Indices>(tup)), 0)... };
}

} // end namespace detail

template <class... T>
bool parse_file(const std::string& path, std::tuple<T...>& tup)
{
    std::ifstream file(path);
    detail::parse_stream(file, tup, std::make_index_sequence<sizeof...(T)>());
    return !file.fail();
}

template <class... T>
bool parse_string(const std::string& param, std::tuple<T...>& tup)
{
    std::istringstream iss(param);
    detail::parse_stream(iss, tup, std::make_index_sequence<sizeof...(T)>());
    return !iss.fail();
}

} // end namespace cpprob
#endif //INCLUDE_SERIALIZATION_HPP
