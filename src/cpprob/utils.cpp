#include "cpprob/utils.hpp"

#include <execinfo.h>
#include <cxxabi.h>

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

boost::random::mt19937& get_rng(){
    static boost::random::mt19937 rng{detail::seeded_rng()};
    return rng;
}

std::string get_addr(){
    constexpr int buf_size = 1000;
    static void *buffer[buf_size];
    char **strings;

    int nptrs = backtrace(buffer, buf_size);

    // We will not store the call to get_traces or the call to sample
    // We discard either observe -> sample_impl -> get_addr
    //            or     sample  -> sample_impl -> get_addr
    std::vector<std::string> trace_addrs;

    strings = backtrace_symbols(buffer, nptrs);
    if (strings == nullptr) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    // Discard calls to sample / observe and subsequent calls
    int i = 0;
    std::string s;
    do {
        if (i == 4){
            std::cerr << "sample_impl or predict not found." << std::endl;
            exit(EXIT_FAILURE);
        }
        s = std::string(strings[i]);
        ++i;
        // CHECK THIS
    } while (s.find("sample_impl") == std::string::npos && s.find("predict") == std::string::npos);
    s = std::string(strings[i]);
    // AND THIS
    if (s.find("cpprob") != std::string::npos &&
        (s.find("sample") != std::string::npos ||
         s.find("observe") != std::string::npos ||
         s.find("predict") != std::string::npos)){
        ++i;
    }

    auto get_name = [](char* s){
        auto str = std::string(s);
        auto first = str.find_last_of('(') + 1;
        auto last = str.find_last_of(')');

        auto mas = str.find_last_of('+');

        // The +3 is to discard the characters
        //auto first = s.find("[0x") + 3;
        //auto last = s.find("]");
        int status;
        char* result = abi::__cxa_demangle(str.substr(first, mas-first).c_str(), nullptr, nullptr, &status);
        auto demangled = std::string(result);
        free(result);
        // Demangled function name + offset w.r.t the function frame
        return demangled + str.substr(mas, last-mas);

        // Mangled option
        //return str.substr(first, last-first);
    };

    // The -4 is to discard the call to _start and the call to __libc_start_main
    // plus the two calls until the function is called
    // plus the two calls to f_default_params, f_default_params_detail
    std::string ret ("[");
    if (i < nptrs - 6){
        ret += get_name(strings[i]);
        ++i;
    }
    for (auto j = i; j < nptrs - 6; j++) {
        ret += ' ' + get_name(strings[j]);
    }
    ret += ']';
    std::free(strings);
    return ret;
}

}  // namespace cpprob
