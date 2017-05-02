#include "cpprob/utils.hpp"

#include <execinfo.h>

#include <cstdlib>
#include <numeric>
#include <iostream>
#include <vector>

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

    size_t nptrs = backtrace(buffer, buf_size);

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
    size_t i = 0;
    std::string s;
    do {
        if (i == 4){
            std::cerr << "sample_impl or predict not found." << std::endl;
            exit(EXIT_FAILURE);
        }
        s = std::string(strings[i]);
        ++i;
    } while (s.find("sample_impl") == std::string::npos && s.find("predict") == std::string::npos);
    s = std::string(strings[i]);
    if (s.find("cpprob") != std::string::npos &&
        (s.find("sample") != std::string::npos ||
         s.find("observe") != std::string::npos ||
         s.find("predict") != std::string::npos)){
        ++i;
    }


    // The -4 is to discard the call to _start and the call to __libc_start_main
    // plus the two calls untill the function is called
    for (size_t j = i; j < nptrs - 4; j++) {
        s = std::string(strings[j]);
        // The +3 is to discard the characters
        auto first = s.find("[0x") + 3;
        auto last = s.find("]");
        trace_addrs.emplace_back(s.substr(first, last-first));
    }
    std::free(strings);
    return std::accumulate(trace_addrs.rbegin(), trace_addrs.rend(), std::string(""));
}

}  // namespace cpprob
