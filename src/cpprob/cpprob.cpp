#include "cpprob.hpp"

#include <string>
#include <unordered_map>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "socket.hpp"
#include "traits.hpp"

namespace cpprob {

std::string get_addr(){
    constexpr int buf_size = 1000;
    static void *buffer[buf_size];
    char **strings;

    size_t nptrs = backtrace(buffer, buf_size);

    // We will not store the call to get_traces or the call to sample
    // We discard either observe -> sample_impl -> get_addr
    //            or     sample  -> sample_impl -> get_addr
    std::vector<std::string> trace;

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
            std::cerr << "sample_impl not found." << std::endl;
            exit(EXIT_FAILURE);
        }
        s = std::string(strings[i]);
        ++i;
    } while (s.find("sample_impl") == std::string::npos);
    s = std::string(strings[i]);
    if (s.find("cpprob") != std::string::npos &&
        (s.find("sample") != std::string::npos || s.find("observe") != std::string::npos)){
        ++i;
    }


    // The -4 is to discard the call to _start and the call to __libc_start_main
    // plus the two calls untill the function is called
    for (size_t j = i; j < nptrs - 4; j++) {
        s = std::string(strings[j]);
        // The +3 is to discard the characters
        auto first = s.find("[0x") + 3;
        auto last = s.find("]");
        trace.emplace_back(s.substr(first, last-first));
    }
    free(strings);
    return std::accumulate(trace.rbegin(), trace.rend(), std::string(""));
}

// Idea from
// http://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device/109266#109266
template<class T = boost::random::mt19937, std::size_t N = T::state_size>
std::enable_if_t<N != 0, T> seeded_rng(){
    std::array<typename T::result_type, N> random_data;
    std::random_device rd;
    std::generate(random_data.begin(), random_data.end(), std::ref(rd));
    std::seed_seq seeds(random_data.begin(), random_data.end());
    return T{seeds};
}

Sample::Sample(int time_index,
               int sample_instance,
               double value,
               const std::string& proposal_name,
               const std::string& sample_address) :
                    time_index_{time_index},
                    sample_instance_{sample_instance},
                    value_{value},
                    proposal_name_{proposal_name},
                    sample_address_{sample_address}{}

void Sample::pack(msgpack::packer<msgpack::sbuffer>& pk) const {
    pk.pack_map(5);
    pk.pack(std::string("time-index"));
    pk.pack(time_index_);
    pk.pack(std::string("proposal-name"));
    pk.pack(proposal_name_);
    pk.pack(std::string("value"));
    pk.pack(value_);
    pk.pack(std::string("sample-instance"));
    pk.pack(sample_instance_);
    pk.pack(std::string("sample-address"));
    pk.pack(sample_address_);
}

Core::Core(bool training, zmq::socket_t* socket) : training_(training), socket_(socket){}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type Core::sample(const Distr<Params ...>& distr) {
    return sample_impl(distr, false);
}

template<template <class ...> class Distr, class ...Params>
void Core::observe(const Distr<Params ...>& distr, double x) {
    if (training_){
        sample_impl(distr, true);
    }
    else{
        using std::log;
        auto prob = pdf(math_distr(distr), x);
        y_.emplace_back(prob);
        w_ += log(prob);
    }
}

void Core::pack(msgpack::packer<msgpack::sbuffer>& pk){
    pk.pack_map(2);
        pk.pack(std::string("samples"));
            pk.pack_array(samples_.size());
            for(const auto& s : samples_)
                s.pack(pk);

        pk.pack(std::string("observes"));
        pk.pack_map(2);
            pk.pack(std::string("shape"));
                pk.pack_array(1);
                pk.pack(observes_.size());
            pk.pack(std::string("data"));
            pk.pack(observes_);
}


template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type Core::sample_distr(const Distr<Params ...>& distr) {
        static boost::random::mt19937 rng{seeded_rng()};
        static boost::random::variate_generator<boost::random::mt19937, Distr<Params ...>> next_val{rng, distr};
        return next_val();
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type Core::sample_impl(const Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = Core::ids_.emplace(addr, static_cast<int>(Core::ids_.size())).first->second;

    if (id >= static_cast<int>(x_.size()))
        x_.resize(id + 1);

    if (training_) {
        static boost::random::mt19937 rng{seeded_rng()};
        static boost::random::variate_generator<boost::random::mt19937, Distr<Params ...>> next_val{rng, distr};
        x = next_val();

        if(from_observe){
            observes_.emplace_back(x);
        }
        else{
            // Lua starts with 1
            samples_.emplace_back(Sample{time_index_, static_cast<int>(x_[id].size()) + 1, x, distr_name<Distr>::value, addr});
        }
    } else {
        // Request client
        // TODO(Lezcano) Use proposal distribution to sample given the parameters from the NN
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);

        // Lua starts with 1
        int sample_instance = x_[id].size() + 1;

        pk.pack_map(2);
        pk.pack(std::string("command"));
        pk.pack(std::string("proposal-params"));

        pk.pack(std::string("command-param"));
        pk.pack_map(6);

        pk.pack(std::string("sample-address"));
        pk.pack(addr);

        pk.pack(std::string("sample-instance"));
        pk.pack(sample_instance);

        pk.pack(std::string("proposal-name"));
        pk.pack(std::string(distr_name<Distr>::value));

        pk.pack(std::string("prev-sample-address"));
        pk.pack(prev_addr_);

        pk.pack(std::string("prev-sample-instance"));
        pk.pack(prev_sample_instance_);

        pk.pack(std::string("prev-sample-value"));
        pk.pack(prev_x_);

        zmq::message_t request (sbuf.size());
        memcpy (request.data(), sbuf.data(), sbuf.size());
        socket_->send(request);

        auto params = receive<std::vector<double>>(*socket_);

        // TODO(Lezcano) Use last_x_ and id to compute x
        // TODO(Lezcano) use std::exchange
        // x = sample_distr(posterior_distr(params));
        x = sample_distr(distr);

        prev_x_ = x;
        prev_sample_instance_ = sample_instance;
        prev_addr_ = addr;


        // TODO(Lezcano) Accumulate log(p/q) where q is the proposal distribution
    }

    x_[id].emplace_back(static_cast<double>(x));

    x_addr_.emplace_back(static_cast<double>(x), id);
    ++time_index_;

    return x;
}

std::unordered_map<std::string, int> cpprob::Core::ids_;

template double Core::sample(const boost::random::normal_distribution<double>&);

template void Core::observe(const boost::random::normal_distribution<double>&, double);
}  // namespace cpprob
