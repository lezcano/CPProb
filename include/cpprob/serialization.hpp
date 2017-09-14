#ifndef INCLUDE_SERIALIZATION_HPP
#define INCLUDE_SERIALIZATION_HPP

#include <array>             // for array
#include <cstddef>           // for size_t
#include <initializer_list>  // for initializer_list
#include <fstream>           // for ifstream
#include <iostream>          // for cerr
#include <map>               // for map
#include <sstream>           // for stringstream
#include <string>            // for operator<<, string
#include <tuple>             // for tuple
#include <utility>           // for make_index_sequence, pair, index_sequence
#include <vector>            // for vector

namespace cpprob {

// Forward declarations
template<class CharT, class Traits, class U, class V>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::pair<U, V>& pair);
template<class CharT, class Traits, class... T>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup);
template<class CharT, class Traits, class T>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::vector<T>& vec);
template<class CharT, class Traits, class T, std::size_t N>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::array<T,N>& vec);
template<class CharT, class Traits, class Key, class Value>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::map<Key,Value>& map);

template<class CharT, class Traits, class U, class V>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits> &os, const std::pair<U, V>& pair);
template<class CharT, class Traits, class... T>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits> &os, const std::tuple<T...>& tup);
template<class CharT, class Traits, class T>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits> &os, const std::vector<T>& vec);
template<class CharT, class Traits, class T, std::size_t N>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits> &os, const std::array<T,N>& vec);
template<class CharT, class Traits, class Key, class Value>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits> &os, const std::map<Key,Value>& map);

template<class CharT, class Traits, class U, class V>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const std::pair<U, V>& pair)
{
    return os << os.widen('(') << pair.first << os.widen(' ') << pair.second << os.widen(')');
}

template<class CharT, class Traits, class... T, size_t... Indices>
void print_tuple_impl(std::basic_ostream<CharT, Traits> &os, const std::tuple<T...>& tup, std::index_sequence<Indices...>)
{
    (void)std::initializer_list<int>
    {
        (os << os.widen(' ') << std::get<Indices+1>(tup)
        , 0)...
    };
}

template<class CharT, class Traits, class... T>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const std::tuple<T...>& tup)
{
    os << os.widen('(');
    if (sizeof...(T) != 0)
    {
        os << std::get<0>(tup);
        print_tuple_impl<CharT, Traits, T...>(os, tup, std::make_index_sequence<sizeof...(T)-1>());
    }
    return os << os.widen(')');
}

template<class CharT, class Traits, class Iter>
std::basic_ostream<CharT, Traits>&
print_iter(std::basic_ostream<CharT, Traits> &os, Iter beg,  Iter end, char init_del, char end_del)
{
    os << os.widen(init_del);
    if (beg != end) {
        os << *beg;
        ++beg;
        for (; beg != end; ++beg) {
            os << os.widen(' ') << *beg;
        }
    }
    return os << os.widen(end_del);
}



template<class CharT, class Traits, class T, std::size_t N>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const std::array<T, N>& arr)
{
    return print_iter(os, arr.cbegin(), arr.cend(), '[', ']');
}

template<class CharT, class Traits, class T>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const std::vector<T>& vec)
{
    return print_iter(os, vec.cbegin(), vec.cend(), '[', ']');
}

template<class CharT, class Traits, class Key, class Value>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits> &os, const std::map<Key,Value>& map)
{
    return print_iter(os, map.cbegin(), map.cend(), '{', '}');
}

template<class CharT, class Traits, class U, class V>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits> &is, std::pair<U, V>& pair)
{
    std::tuple<U, V> tup;
    is >> tup;
    if (is.fail())
        return is;
    pair.first = std::move(std::get<0>(tup));
    pair.second = std::move(std::get<1>(tup));
    return is;
}

template<class CharT, class Traits, class... T, size_t... Indices>
void read_tuple_impl(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup, std::index_sequence<Indices...>)
{
    (void)std::initializer_list<int>{ (is >> std::get<Indices>(tup), 0)... };
}

template<class CharT, class Traits, class... T>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits> &is, std::tuple<T...>& tup) {
    CharT ch;
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen('(')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return is;
    }

    // read_tuple_impl will read exactly the number of elements of the tuple
    // If the is.fail() is set it is because there was an error
    read_tuple_impl<CharT, Traits, T...>(is, tup, std::make_index_sequence<sizeof...(T)>());
    if (is.fail())
        return is;

    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen(')')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
    return is;
}

template<class CharT, class Traits, class T>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits> &is, std::vector<T>& vec)
{
    CharT ch;
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return is;
    }

    do {
        // Needed to be declared inside the loop, otherwise std::move
        // could leave it in an invalid state
        T val;
        is >> val;
        if(is.fail())
            break;
        vec.emplace_back(std::move(val));
    } while (true);

    // Remark: This accepts vectors not properly specified like
    // [(1 2) (1 4] for std::vector<std::pair<int, int>> or [] for std::vector<std::vector<T>>
    // but well...
    is.clear();
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen(']')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
    return is;
}

template<class CharT, class Traits, class T, std::size_t N>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits> &is, std::array<T, N>& arr)
{
    CharT ch;
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return is;
    }

    for (auto & elem : arr) {
        if (!(is >> elem))
            return is;
    }
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen(']')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
    return is;
}

template<class CharT, class Traits, class Key, class Value>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits> &is, std::map<Key,Value>& map)
{
    CharT ch;
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen('{')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return is;
    }

    do {
        // Needed to be declared inside the loop, otherwise std::move
        // could leave it in an invalid state
        typename std::map<Key, Value>::value_type val;
        is >> val;
        if(is.fail())
            break;
        map.insert(std::move(val));
    } while (true);

    // Remark: This accepts vectors not properly specified like
    // [(1 2) (1 4] for std::vector<std::pair<int, int>> or [] for std::vector<std::vector<T>>
    // but well...
    is.clear();
    if (!(is >> std::ws >> ch)) {
        return is;
    }
    if (ch != is.widen('}')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
    }
    return is;

}

template<class... T, class CharT, class Traits, size_t... Indices>
void parse_stream(std::basic_istream<CharT, Traits>& is, std::tuple<T...>& tup, std::index_sequence<Indices ...>)
{
    (void)std::initializer_list<int>{ (is >> std::get<Indices>(tup), 0)... };
}

template <class... T>
bool parse_file(const std::string& path, std::tuple<T...>& tup)
{
    std::ifstream file(path);
    if (file) {
        parse_stream(file, tup, std::make_index_sequence<sizeof...(T)>());
    }
    else {
        std::cerr << "File " << path << " could not be opened.\n";
    }
    return !file.fail();
}

template <class... T>
bool parse_string(const std::string& param, std::tuple<T...>& tup)
{
    std::istringstream iss(param);
    parse_stream(iss, tup, std::make_index_sequence<sizeof...(T)>());
    return !iss.fail();
}
} // end namespace cpprob
#endif //INCLUDE_SERIALIZATION_HPP
