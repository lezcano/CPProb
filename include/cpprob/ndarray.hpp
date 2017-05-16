#ifndef INCLUDE_NDARRAY_HPP
#define INCLUDE_NDARRAY_HPP

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpprob/serialization.hpp"

namespace cpprob {

template<class T = double>
class NDArray {
public:
    // Iterator utilities
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    NDArray() = default;

    NDArray(T a) : values_{a}, shape_{1} {}

    template<class U,
            class = std::enable_if_t<std::is_constructible<T, U>::value>>
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
    NDArray(const std::vector<U> &v)
    {
        compute_shape(v, 0);
        values_ = flatten(v, 0);
    }

    // Iterator utilities
    iterator begin() { return values_.begin(); }
    const_iterator begin() const { return values_.begin(); }
    iterator end() { return values_.end(); }
    const_iterator end() const { return values_.end(); }
    const_iterator cbegin() const { return values_.cbegin(); }
    const_iterator cend() const { return values_.cend(); }

    NDArray& operator+=(const NDArray & rhs)
    {
        if (this->shape().empty()){
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
                       std::plus<T>());
        return *this;
    }

    NDArray& operator-=(const NDArray & rhs)
    {
        if (this->shape().empty()){
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
                       std::minus<T>());
        return *this;
    }

    NDArray& operator*=(const NDArray & rhs)
    {
        if (this->shape().empty()){
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
                       std::multiplies<T>());
        return *this;
    }

    NDArray& operator*=(const T rhs)
    {
        std::transform(values_.begin(),
                       values_.end(),
                       values_.begin(),
                       [rhs](T a){ return rhs * a; });
        return *this;
    }

    NDArray& operator/=(const T rhs)
    {

        std::transform(values_.begin(),
                       values_.end(),
                       values_.begin(),
                       [rhs](T a){ return a / rhs; });
        return *this;
    }

    // friend functions
    friend NDArray operator+ (const NDArray& lhs, const NDArray& rhs){ return NDArray(lhs) += rhs; }
    friend NDArray operator- (const NDArray& lhs, const NDArray& rhs){ return NDArray(lhs) -= rhs; }
    friend NDArray operator* (const NDArray& lhs, const NDArray& rhs){ return NDArray(lhs) *= rhs; }
    friend NDArray operator* (const T lhs, const NDArray& rhs){ return NDArray(rhs) *= lhs; }
    friend NDArray operator* (const NDArray& lhs, const T rhs){ return NDArray(lhs) *= rhs; }

    friend NDArray<T> sqrt (const NDArray<T> & arr)
    {
        std::vector<T> ret(arr.values_.size());
        std::transform(arr.values_.begin(), arr.values_.end(), ret.begin(), [](T elem) { return std::sqrt(elem); });
        return NDArray<T>(ret, arr.shape());
    }

    friend NDArray<T> log (const NDArray<T> & arr)
    {
        std::vector<T> ret(arr.values_.size());
        std::transform(arr.values_.begin(), arr.values_.end(), ret.begin(), [](T elem) { return std::log(elem); });
        return NDArray<T>(ret, arr.shape());
    }

    friend NDArray<T> exp (const NDArray<T> & arr)
    {
        std::vector<T> ret(arr.values_.size());
        std::transform(arr.values_.begin(), arr.values_.end(), ret.begin(), [](T elem) { return std::exp(elem); });
        return NDArray<T>(ret, arr.shape());
    }

    template<class Iter, class = std::enable_if_t<
                                        std::is_same<
                                                typename std::iterator_traits<Iter>::value_type,
                                                NDArray<T>>::value >>
    friend NDArray<T> supremum (Iter begin, Iter end)
    {
        using namespace cpprob::detail; // print utilities
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
            std::transform(begin.begin(), begin.end(), ret.begin(), ret.begin(), [](T a, T b){ return std::max(a, b); });
            ++begin;
        }
        return ret;
    }

    template<class CharT, class Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const NDArray & v)
    {
        using namespace detail; // operator << for containers
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
        using namespace detail; // operator >> for containers
        CharT ch;
        T scalar;

        if(!(is >> std::ws >> ch)){
            return is;
        }

        // Scalar
        if(ch != is.widen('[')){
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
    template<class U,
            class = std::enable_if_t<std::is_constructible<T, U>::value>>
    void compute_shape(const std::vector<U> & v, int i)
    {
        if (static_cast<int>(shape_.size()) <= i)
            shape_.emplace_back(v.size());
        else if (shape_[i] < static_cast<int>(v.size()))
            shape_[i] = v.size();
    }

    // i is the dimension that we are processing
    template<class U>
    void compute_shape(const std::vector<std::vector<U>> & v, int i)
    {
        if (static_cast<int>(shape_.size()) <= i)
            shape_.emplace_back(v.size());
        else if (static_cast<int>(v.size()) > shape_[i])
            shape_[i] = v.size();
        for (const auto& e : v)
            compute_shape(e, i+1);
    }

    template<class U,
            class = std::enable_if_t<std::is_constructible<T, U>::value>>
    std::vector<T> flatten(const std::vector<U> & v, int)
    {
        auto ret = v;
        ret.resize(shape_.back());
        return std::vector<T>(std::make_move_iterator(ret.begin()), std::make_move_iterator(ret.end()));
    }

    template<class U>
    auto flatten(const std::vector<std::vector<U>> & v, int i)
    {
        auto size_tensor_i = std::accumulate(shape_.begin()+i, shape_.end(), 1, std::multiplies<int>());

        decltype(flatten(v.front(), i+1)) ret;
        for (const auto &e : v) {
            auto init_e = flatten(e, i+1);
            ret.insert(ret.end(), std::make_move_iterator(init_e.begin()), std::make_move_iterator(init_e.end()));
        }
        ret.resize(size_tensor_i);
        return ret;
    }

    std::vector<T> values_;
    std::vector<int> shape_;
};
} // end namespace cpprob
#endif //INCLUDE_NDARRAY_HPP
