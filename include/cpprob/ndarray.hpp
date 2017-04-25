#ifndef INCLUDE_NDARRAY_HPP
#define INCLUDE_NDARRAY_HPP

#include <vector>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <iostream>

#include "cpprob/detail/vector_io.hpp"

namespace cpprob {

template<class T = double>
class NDArray {
public:
    NDArray() = default;

    NDArray(T a) : values_{a}, shape_{1} {}

    template<class U,
            class = std::enable_if_t <std::is_constructible<T, U>::value>>
    NDArray(U a) : NDArray(T(a)) {};

    template<class U>
    NDArray(const std::vector <U> &v)  : values_(init(v))
    {
        shape_.insert(shape_.begin(), v.size());
    }

    NDArray& operator+=(const NDArray & rhs)
    {
        if (this->shape() != rhs.shape())
            throw std::domain_error("The tensors do not have the same shape");

        std::transform(values_.begin(),
                       values_.end(),
                       rhs.values_.begin(),
                       values_.begin(),
                       std::plus<double>());
        return *this;
    }

    NDArray& operator*=(const double rhs)
    {
        std::transform(values_.begin(),
                       values_.end(),
                       values_.begin(),
                       [rhs](double a){ return rhs * a; });
        return *this;
    }

    NDArray& operator/=(const double rhs)
    {

        std::transform(values_.begin(),
                       values_.end(),
                       values_.begin(),
                       [rhs](double a){ return a / rhs; });
        return *this;
    }

    // friend functions
    friend NDArray operator+ (const NDArray& lhs, const NDArray& rhs){ return NDArray(lhs) += rhs; }
    friend NDArray operator* (const double lhs, const NDArray& rhs){ return NDArray(rhs) *= lhs; }
    friend NDArray operator* (const NDArray& lhs, const double rhs){ return NDArray(lhs) *= rhs; }

    template<class CharT, class Traits>
    friend std::basic_ostream< CharT, Traits > &
    operator<<(std::basic_ostream< CharT, Traits > & os, const NDArray & v) {
        if (v.shape_.size() == 1 && v.shape_[0] == 1) {
            os << v.values_[0];
            return os;
        }

        detail::print_vector(os, v.values_);

        if (v.shape_.size() != 1) {
            os << os.widen(' ');
            detail::print_vector(os, v.shape_);
        }
        return os;
    }

    std::vector <T> values() const
    {
        return values_;
    }

    std::vector <int> shape() const
    {
        return shape_;
    }

private:

    template<class U,
            class = std::enable_if_t <std::is_constructible<T, U>::value>>
    std::vector <T> init(const std::vector <U> &v)
    {
        return v;
    }

    template<class U>
    std::vector <U> init(const std::vector <std::vector<U>> &v)
    {
        std::vector <std::vector<U>> aux;
        for (const auto &e : v) {
            aux.emplace_back(init(e));
        }
        auto max_size = (*std::max_element(aux.begin(), aux.end(),
                                           [](std::vector <U> v1, std::vector <U> v2) {
                                               return v1.size() < v2.size();
                                           })).size();
        std::vector <U> ret;
        for (auto &e : aux) {
            e.resize(max_size);
            ret.insert(ret.end(), std::make_move_iterator(e.begin()), std::make_move_iterator(e.end()));
        }
        shape_.insert(shape_.begin(), max_size);
    }

    // The order is important!!
    std::vector <int> shape_;
    std::vector <T> values_;
};
} // end namespace cpprob
#endif //INCLUDE_NDARRAY_HPP
