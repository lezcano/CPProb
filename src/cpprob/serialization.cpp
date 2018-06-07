#include "cpprob/serialization.hpp"

// TODO(Lezcano) Hack!!
// Otherwise type erasue in cpprob::any does not instantiante the templates
namespace cpprob {
    template
    std::basic_ostream<char, std::char_traits<char>>& operator<<(std::basic_ostream<char, std::char_traits<char>> &os, const std::vector<std::vector<std::vector<double>>>& vec);
}
