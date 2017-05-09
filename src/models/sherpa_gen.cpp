#include <iostream>

#include "models/sherpa_mini.hpp"
#include "cpprob/serialization.hpp"

int main(){
    cpprob::detail::print(std::cout, models::dummy_sherpa());
}
