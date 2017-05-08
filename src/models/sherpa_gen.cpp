#include <iostream>

#include "models/sherpa_mini.hpp"
#include "cpprob/detail/vector_io.hpp"

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    out << "[ ";
    for (const auto &elem : v) {
        out << elem << " ";
    }
    out << "]";
    return out;
}

int main(){
    std::cout << models::dummy_sherpa() << std::endl;
}
