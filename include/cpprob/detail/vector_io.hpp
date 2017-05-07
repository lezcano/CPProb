#ifndef INCLUDE_DETAIL_VECTOR_IO_HPP
#define INCLUDE_DETAIL_VECTOR_IO_HPP

#include <iostream>
#include <vector>

namespace cpprob {

namespace detail {

template<class CharT, class Traits, class T>
void print_vector(std::basic_ostream <CharT, Traits> &os,
                  const std::vector <T> &vec) {
    auto iter = vec.begin(), end = vec.end();
    os << os.widen('[');
    if (iter != end) {
        os << *iter;
        ++iter;
        for (; iter != end; ++iter) {
            os << os.widen(' ') << *iter;
        }
    }
    os << os.widen(']');
}

template<class CharT, class Traits, class T>
void read_vector(std::basic_istream <CharT, Traits> &is, std::vector <T> &vec) {
    CharT ch;
    if (!(is >> ch)) {
        return;
    }
    if (ch != is.widen('[')) {
        is.putback(ch);
        is.setstate(std::ios_base::failbit);
        return;
    }
    T val;
    while (is >> std::ws >> val) {
        vec.push_back(val);
    }
    if (is.fail()) {
        is.clear();
        if (!(is >> ch)) {
            return;
        }
        if (ch != is.widen(']')) {
            is.putback(ch);
            is.setstate(std::ios_base::failbit);
        }
    }
}

} // end namespace detail
} // end namespace cpprob
#endif // INCLUDE_DETAIL_VECTOR_IO_HPP
