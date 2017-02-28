#include <unordered_map>
#include <string>

#include "cpprob.hpp"

template<bool Inference>
std::unordered_map<std::string, int> cpprob::Core<Inference>::ids_;
