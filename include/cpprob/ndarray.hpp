#ifndef INCLUDE_NDARRAY_HPP
#define INCLUDE_NDARRAY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>  // std::bad_cast
#include <type_traits>
#include <utility>
#include <vector>

#include "cpprob/serialization.hpp"
#include "cpprob/traits.hpp"

namespace cpprob {

template<class T = double>
class NDArray {
public:
    // Iterator utilities
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    NDArray() = default;

    NDArray(T a) : values_{a}, shape_{1} {}

    template<class U, std::enable_if_t<std::is_constructible<T, U>::value, int> = 0>
    NDArray(U a) : NDArray(T(a)) {}

    NDArray(std::vector<T> values, std::vector<int> shape) : values_(std::move(values)), shape_(std::move(shape))
    {
        #ifndef NDEBUG
        // Check that dimensions agree if values is not empty.
        // If values is empty it might be a default initialisation
        if (!values_.empty()) {
            auto a = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
            if (a != static_cast<int>(values_.size()))
                throw std::runtime_error("Product of the elements of the shape vector " + std::to_string(a) +
                                         " is not equal to the size of the values vector " +
                                         std::to_string(values_.size()) + ".");
        }
        #endif
    }

    template<class U>
    NDArray(const std::vector<U> & values) : NDArray(values.begin(), values.end()) {}

    template<class U, std::size_t N>
    NDArray(const std::array<U, N> & values) : NDArray(values.begin(), values.end()) {}

    template<class Iter>
    NDArray(Iter begin, Iter end)
    {
        compute_shape(begin, end, 0);
        values_ = flatten(begin, end, 0);
    }

    // Casting to a different inner type
    template<class U,
             std::enable_if_t<std::is_constructible<T, U>::value, int> = 0>
    explicit operator NDArray<U>() const
    {
        return NDArray<U>(std::vector<U>(values_.begin(), values_.end()), shape_);
    }

    bool is_vector() const
    {
        return shape_.size() == 1;
    }

    bool is_scalar() const
    {
        return is_vector() && shape_.front() == 1;
    }

    // Casting for scalars
    explicit operator T() const
    {
        if (!is_scalar()) {
            throw std::bad_cast();
        }
        return values_.front();
    }

    // Casting for vectors
    explicit operator std::vector<T>() const
    {
        if (!is_vector()) {
            throw std::bad_cast();
        }
        return values_;
    }

    // Iterator utilities
    iterator begin() { return values_.begin(); }
    const_iterator begin() const { return values_.begin(); }
    iterator end() { return values_.end(); }
    const_iterator end() const { return values_.end(); }
    const_iterator cbegin() const { return values_.cbegin(); }
    const_iterator cend() const { return values_.cend(); }

    // Comparison operators
    friend bool operator==(const NDArray & lhs, const NDArray & rhs )
    {
        return std::tie(lhs.shape_, lhs.values_) == std::tie(rhs.shape_, rhs.values_);
    }


    friend bool operator<(const NDArray & lhs, const NDArray & rhs )
    {
        return std::tie(lhs.shape_, lhs.values_) < std::tie(rhs.shape_, rhs.values_);
    }

    friend bool operator>(const NDArray & lhs, const NDArray & rhs )
    {
        return std::tie(lhs.shape_, lhs.values_) > std::tie(rhs.shape_, rhs.values_);
    }

    friend bool operator!=(const NDArray & lhs, const NDArray & rhs ) { return !(lhs == rhs); }
    friend bool operator<=(const NDArray & lhs, const NDArray & rhs ) { return lhs < rhs || lhs == rhs; }
    friend bool operator>=(const NDArray & lhs, const NDArray & rhs ) { return lhs > rhs || lhs == rhs; }

    // Arithmetic operators
    // In-place operators
    NDArray & operator+=(const NDArray & rhs) { return map_binary(rhs, std::plus<T>()); }
    NDArray & operator-=(const NDArray & rhs) { return map_binary(rhs, std::minus<T>()); }
    NDArray & operator*=(const NDArray & rhs) { return map_binary(rhs, std::multiplies<T>()); }
    NDArray & operator/=(const NDArray & rhs) { return map_binary(rhs, std::divides<T>()); }

    NDArray & operator+=(const T rhs) { return map_binary(rhs, [](T & lhs, const T rhs) { lhs += rhs; }); }
    NDArray & operator-=(const T rhs) { return map_binary(rhs, [](T & lhs, const T rhs) { lhs -= rhs; }); }
    NDArray & operator*=(const T rhs) { return map_binary(rhs, [](T & lhs, const T rhs) { lhs *= rhs; }); }
    NDArray & operator/=(const T rhs) { return map_binary(rhs, [](T & lhs, const T rhs) { lhs /= rhs; }); }

    // Friend operators
    // Binary
    friend NDArray operator+(const NDArray & lhs, const NDArray & rhs) { return NDArray(lhs) += rhs; }
    friend NDArray operator+(const T lhs, const NDArray & rhs) { return NDArray(rhs) += lhs; }
    friend NDArray operator+(const NDArray & lhs, const T rhs) { return NDArray(lhs) += rhs; }

    friend NDArray operator-(const NDArray & lhs, const NDArray & rhs) { return NDArray(lhs) -= rhs; }
    friend NDArray operator-(const NDArray & lhs, const T rhs) { return NDArray(lhs) -= rhs; }

    friend NDArray operator*(const NDArray & lhs, const NDArray & rhs) { return NDArray(lhs) *= rhs; }
    friend NDArray operator*(const T lhs, const NDArray & rhs) { return NDArray(rhs) *= lhs; }
    friend NDArray operator*(const NDArray & lhs, const T rhs) { return NDArray(lhs) *= rhs; }

    friend NDArray operator/(const NDArray & lhs, const NDArray & rhs) { return NDArray(lhs) /= rhs; }
    friend NDArray operator/(const NDArray & lhs, const T rhs) { return NDArray(lhs) /= rhs; }

    // Unary
    NDArray<T> sqrt() { map_unary(&std::sqrt); }
    NDArray<T> exp () { map_unary(&std::exp); }
    NDArray<T> log () { map_unary(&std::log); }


    template<class Iter, std::enable_if_t<
                                std::is_same<
                                        typename std::iterator_traits<Iter>::value_type,
                                        NDArray<T>
                                >::value, int> = 0>
    friend NDArray<T> supremum (Iter begin, Iter end)
    {
        if (begin == end) {
            return NDArray<T>();
        }
        NDArray<T> ret = *begin;
        ++begin;

        while(begin != end) {
            if (ret.shape() != begin->shape()) {
                std::ostringstream ss;
                ss << "Error, the tensors of shape " << ret.shape()
                   << " and " << begin->shape() << " are not compatible.\n";
                throw std::range_error(ss.str());
            }
            std::transform(begin.begin(), begin.end(), ret.begin(), ret.begin(), [](T a, T b) { return std::max(a, b); });
            ++begin;
        }
        return ret;
    }

    template<class CharT>
    friend std::basic_ostream<CharT> &
    operator<<(std::basic_ostream<CharT> & os, const NDArray & v)
    {
        if (v.shape_.size() == 1 && v.shape_[0] == 1) {
            os << v.values_[0];
            return os;
        }

        os << v.values_;

        if (v.shape_.size() > 1) { // Is a 2-tensor or higher
            os << os.widen(' ') << os.widen('s') << v.shape_;
        }
        return os;
    }

    template<class CharT, class Traits>
    friend std::basic_istream< CharT, Traits > &
    operator>>(std::basic_istream< CharT, Traits > & is, NDArray & v)
    {
        CharT ch;
        T scalar;

        if(!(is >> std::ws >> ch)) {
            return is;
        }

        // Scalar
        if(ch != is.widen('[')) {
            is.putback(ch);
            if(!(is >> std::ws >> scalar))
            {
                return is;
            }
            v.values_ = std::vector<T>{scalar};
            v.shape_ = std::vector<int>{1};
            return is;
        }

        // Vector or tensor
        is.putback(ch);
        is >> v.values_;

        // It might be a tensor if we find a shape of the form s[a1 a2 ... an]
        // if not, it is a vector
        if(!(is >> std::ws >> ch)) {
            is.clear();
            is.putback(ch);
            v.shape_ = std::vector<int>{static_cast<int>(v.values_.size())};
            return is;
        }

        if (ch != is.widen('s')) {
            is.putback(ch);
            is.setstate(std::ios_base::failbit);
            return is;
        }
        is >> v.shape_;

        return is;
    }

    friend NDArray<T> get_zero(const NDArray<T> & x)
    {
        return NDArray<T>(std::vector<T>(x.values_.size(), 0), x.shape());
    }

    std::size_t size() const
    {
        return values_.size();
    }

    std::vector<T> values() const
    {
        return values_;
    }

    std::vector<int> shape() const
    {
        return shape_;
    }

private:
    // Base type
    template<class Iter, std::enable_if_t<std::is_constructible<T, typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    void compute_shape(Iter begin, Iter end, int depth)
    {
        const auto size = std::distance(begin, end);
        // If it's the first time that we enter the function, we extend the size
        // If not, we update it
        if (static_cast<int>(shape_.size()) <= depth) {
            shape_.emplace_back(size);
        }
        else if (shape_[depth] < static_cast<int>(size)) {
            shape_[depth] = size;
        }
    }

    // TODO(Lezcano) Duplicated code
    template<class Iter, std::enable_if_t<is_tuple_like<typename std::iterator_traits<Iter>::value_type>::value &&
                                         !is_iterable<typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    void compute_shape(Iter begin, Iter end, int depth)
    {
        auto size = std::distance(begin, end);
        // If it's the first time that we enter the function, we extend the size
        // If not, we update it
        if (static_cast<int>(shape_.size()) <= depth) {
            shape_.emplace_back(size);
        }
        else if (shape_[depth] < static_cast<int>(size)) {
            shape_[depth] = size;
        }
        depth++;
        size = std::tuple_size<typename std::iterator_traits<Iter>::value_type>::value;
        if (static_cast<int>(shape_.size()) <= depth) {
            shape_.emplace_back(size);
        }
        else if (shape_[depth] < static_cast<int>(size)) {
            shape_[depth] = size;
        }
    }

    // Iterable type
    // Check if the Iter type corresponds to an iterator that points to an iterable object
    template<class Iter, std::enable_if_t< is_iterable<typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    void compute_shape(Iter begin, Iter end, int depth)
    {
        auto size = std::distance(begin, end);
        if (static_cast<int>(shape_.size()) <= depth)
            shape_.emplace_back(size);
        else if (static_cast<int>(size) > shape_[depth])
            shape_[depth] = size;
        for (; begin != end; ++begin)
            compute_shape(std::begin(*begin), std::end(*begin), depth+1);
    }

    template<class Iter, std::enable_if_t< std::is_constructible<T, typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    std::vector<T> flatten(Iter begin, Iter end, int)
    {
        std::vector<T> ret (begin, end);
        ret.resize(shape_.back());
        return ret;
    }

    // TODO(Lezcano) Duplicate code!! if constexpr would be nice
    template<class Iter, std::enable_if_t<is_tuple_like<typename std::iterator_traits<Iter>::value_type>::value &&
                                          !is_iterable<typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    std::vector<T> flatten(Iter begin, Iter end, int depth)
    {
        const auto size_tensor_i = std::accumulate(shape_.begin()+depth, shape_.end(), 1, std::multiplies<int>());

        std::vector<T> ret;
        for (; begin != end; ++begin) {
            auto init_e = flatten(*begin);
            ret.insert(ret.end(), std::make_move_iterator(init_e.begin()), std::make_move_iterator(init_e.end()));
        }
        ret.resize(size_tensor_i);
        return ret;
    }

    template<class Iter, std::enable_if_t<is_iterable<typename std::iterator_traits<Iter>::value_type>::value, int> = 0>
    std::vector<T> flatten(Iter begin, Iter end, int depth)
    {
        const auto size_tensor_i = std::accumulate(shape_.begin()+depth, shape_.end(), 1, std::multiplies<int>());

        std::vector<T> ret;
        for (; begin != end; ++begin) {
            auto init_e = flatten(std::begin(*begin), std::end(*begin), depth+1);
            ret.insert(ret.end(), std::make_move_iterator(init_e.begin()), std::make_move_iterator(init_e.end()));
        }
        ret.resize(size_tensor_i);
        return ret;
    }

    template<class Tup, std::size_t... Indices>
    std::vector<T> flatten_impl(const Tup & tup, std::index_sequence<Indices...>)
    {
        return std::vector<T>{std::get<Indices>(tup)...};
    }

    template<class Tup, std::enable_if_t<is_tuple_like<Tup>::value, int> = 0>
    std::vector<T> flatten(const Tup & tup)
    {
        return flatten_impl(tup, std::make_index_sequence<std::tuple_size<Tup>::value>{});
    }


    template<class F>
    NDArray & map_binary(const NDArray & rhs, const F & f)
    {
        if (this->shape().empty()) {
            *this = rhs;
            return *this;
        }
        if (rhs.shape_.empty())
            return *this;
        if (this->shape() != rhs.shape())
            throw std::domain_error("The tensors do not have the same shape");

        std::transform(values_.begin(),
                       values_.end(),
                       rhs.values_.begin(),
                       values_.begin(),
                       f);
        return *this;
    }

    template<class F>
    NDArray & map_binary(const T rhs, const F & f)
    {
        for (auto & elem : values_) {
            f(elem, rhs);
        }
        return *this;
    }

    template<class F>
    NDArray<T> map_unary(const F & f)
    {
        NDArray<T> ret = *this;
        for (auto & elem : ret.values_) {
            elem = f(elem);
        }
        return ret;
    }

    std::vector<T> values_;
    std::vector<int> shape_;
};

} // end namespace cpprob
#endif //INCLUDE_NDARRAY_HPP
