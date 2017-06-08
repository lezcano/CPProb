#include <iostream>
#include <cpprob/state.hpp>

#include "cpprob/serialization.hpp"
#include "models/sherpa_mini.hpp"

int main(){
    using namespace cpprob;
    State::set(StateType::dryrun);
    std::cout << models::sherpa_mini() << std::endl;
}
