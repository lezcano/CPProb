#include <iostream>
#include <cpprob/state.hpp>

#include "cpprob/serialization.hpp"
#include "models/sherpa_mini.hpp"

int main(){
    using namespace cpprob::detail; // I/O vectors
    cpprob::State::set(cpprob::StateType::dryrun);
    std::cout << cpprob::models::sherpa_mini() << std::endl;
}
