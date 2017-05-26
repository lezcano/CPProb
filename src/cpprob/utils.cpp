#include "cpprob/utils.hpp"

#include <execinfo.h>
#include <cxxabi.h>
#include <cstdio>
#include <cstdlib>

#include <cstdlib>
#include <numeric>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include <algorithm>

#include <boost/random/random_device.hpp>

namespace cpprob{

boost::random::mt19937& get_rng()
{
    static boost::random::mt19937 rng{detail::seeded_rng()};
    return rng;
}

std::string get_name_mangled (char* s)
{
    auto str = std::string(s);
    auto first = str.find_last_of('(') + 1;
    auto last = str.find_last_of(')');

    return str.substr(first, last-first);
}

std::string get_name_demangled (char* s)
{
    auto str = std::string(s);
    auto first = str.find_last_of('(') + 1;
    auto last = str.find_last_of(')');

    auto mas = str.find_last_of('+');

    int status;
    char* result = abi::__cxa_demangle(str.substr(first, mas-first).c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        auto demangled = std::string(result);
        free(result);
        // Demangled function name + offset w.r.t the function frame
        return demangled + str.substr(mas, last - mas);
    }
    else {
        return "System call";
    }
}


std::string get_addr()
{
    constexpr int buf_size = 100;
    static void *buffer[buf_size];
    char **strings;

    int nptrs = backtrace(buffer, buf_size);

    // We will not store the call to get_traces or the call to sample
    // We discard either observe -> sample_impl -> get_addr
    //            or     sample  -> sample_impl -> get_addr
    std::vector<std::string> trace_addrs;

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == nullptr) {
        std::perror("backtrace_symbols");
        std::exit(EXIT_FAILURE);
    }

    // We will consider addresses in the range [begin, end]
    int begin = -1, end = nptrs;
    std::string s;

    // Discard calls inside the cpprob library
    do {
        ++begin;
        s = get_name_demangled(strings[begin]);
    } while (s.find("cpprob::") != std::string::npos && begin != nptrs);

    if (begin == nptrs){
        std::cerr << "Entry function call to the cpprob library not found" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Discard calls until the model is hit
    do {
        --end;
        s = get_name_demangled(strings[end]);
    } while (s.find("models::") == std::string::npos && end != 0);

    if (end == 0){
        std::cerr << "Entry call to model not found. Is the model in the namespace models?" << std::endl;
        std::exit(EXIT_FAILURE);
    }


    std::string ret = "[";
    // Stack calls are reversed. strings[0] is get_addr()
    if (begin <= end) {
        ret += get_name_demangled(strings[end]);
        --end;
    }
    for (auto i = end; i >= begin; i--){
        ret += ' ' + get_name_demangled(strings[i]);
    }
    ret += ']';
    std::free(strings);
    return ret;
}

}  // namespace cpprob
