#include "cpprob/utils.hpp"

#include <cxxabi.h>    // for __cxa_demangle
#include <execinfo.h>  // for backtrace, backtrace_symbols

#include <cstdio>      // for perror
#include <cstdlib>     // for exit, free, EXIT_FAILURE
#include <iostream>    // for cerr
#include <random>      // for mt19937
#include <regex>       // for regex_search, regex, smatch
#include <string>      // for string, basic_string, operator+
#include <vector>      // for vector

namespace cpprob{

std::mt19937& get_rng()
{
    static std::mt19937 rng{detail::seeded_rng()};
    return rng;
}

std::string get_name_mangled (const char* s)
{
    auto str = std::string(s);
    auto first = str.find_last_of('(') + 1;
    auto last = str.find_last_of(')');

    return str.substr(first, last-first);
}

std::string get_name_demangled (const char* s)
{
    auto str = std::string(s);
    auto first = str.find_last_of('(') + 1;
    auto last = str.find_last_of(')');
    auto mas = str.find_last_of('+');

    int status;
    char* result = abi::__cxa_demangle(str.substr(first, mas-first).c_str(), nullptr, nullptr, &status);
    if (status == 0) {
        auto demangled = std::string(result);
        std::free(result);
        // Demangled function name + offset w.r.t the function return address
        return demangled + str.substr(mas, last - mas);
    }
    else {
        return get_name_mangled(s);
    }
}

bool in_namespace_models(const std::string & fun)
{
    // The only spaces before a word in a demangled name are in the arguments and in the name function
    // You also have spaces before a > when another > precedes it, but it's alright
    static std::regex r{"[^,] models::", std::regex::optimize};
    static std::smatch match;
    // TODO(Lezcano) Hack to deal with non templated functions
    // abi::__cxa_demangle does not add the return type when the function is not a template!!
    return std::regex_search(fun, match, r) || fun.find("models::") == 0;
}

bool in_namespace_cpprob(const std::string & fun)
{
    // The only spaces before a word in a demangled name are in the arguments and in the name function
    // You also have spaces before a > when another > precedes it, but it's alright
    static std::regex r{"[^,] cpprob::", std::regex::optimize};
    static std::smatch match;
    // abi::__cxa_demangle does not add the return type when the function is not a template!!
    return std::regex_search(fun, match, r) || fun.find("cpprob::") == 0;
}

std::string get_addr() {
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

    int begin = 0, end = nptrs - 1;

    // Discard calls inside the cpprob library
    while (begin != nptrs && in_namespace_cpprob(get_name_demangled(strings[begin]))) {
        ++begin;
    }

    if (begin == nptrs){
        std::cerr << "Entry function call to the cpprob library not found" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Discard calls outside of the models namespace
    while (end != -1 && !in_namespace_models(get_name_demangled(strings[end]))) {
        --end;
    }
    #ifdef BUILD_SHERPA
    // HACK Discard SherpaWrapper::operator() and SherpaWrapper::sherpa()
    // so we don't have to recompile if we make changes in it
    end -= 2;
    #endif

    if (end == -1){
        std::cerr << "Entry call to model not found. Is the model in the namespace models?" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string ret = "[";
    // Stack calls are reversed. strings[0] is the current function call get_addr()
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
