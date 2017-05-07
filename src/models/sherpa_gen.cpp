#include <iostream>

#include "models/sherpa_mini.hpp"
#include "cpprob/detail/vector_io.hpp"


int main(){
    cpprob::detail::print_vector(std::cout, models::dummy_sherpa());
    std::cout << std::endl;
}