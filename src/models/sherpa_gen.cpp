#include <iostream>

#include "cpprob/serialization.hpp"
#include "models/sherpa_mini.hpp"

int main(){
    using namespace cpprob::detail; // I/O vectors
    std::cout << cpprob::models::dummy_sherpa() << std::endl;
}
