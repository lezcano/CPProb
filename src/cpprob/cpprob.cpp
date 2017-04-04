#include "cpprob.hpp"

#include <string>
#include <unordered_map>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "socket.hpp"
#include "traits.hpp"
#include "trace.hpp"

namespace cpprob {

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

// Static variables
zmq::socket_t* socket = nullptr;
Trace t;
bool training;
boost::random::mt19937 rng{seeded_rng()};
std::unordered_map<std::string, int> ids;
PrevSampleInference prev_sample;
SampleInference curr_sample;


void set_socket(zmq::socket_t* s){ socket = s; }

void reset_trace(){
    t = Trace();
    prev_sample = PrevSampleInference();
    curr_sample = SampleInference();
}

Trace get_trace(){ return t; }

void set_training(const bool t){ training = t; }

void reset_ids(){ ids.clear(); }

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
        trace_addrs.emplace_back(s.substr(first, last-first));
    }
    free(strings);
    return std::accumulate(trace_addrs.rbegin(), trace_addrs.rend(), std::string(""));
}

void pack_request(msgpack::packer<msgpack::sbuffer>& pk) {
    pk.pack_map(2);
    pk.pack(std::string("command"));
    pk.pack(std::string("proposal-params"));

    pk.pack(std::string("command-param"));
    pk.pack_map(6);

    pk.pack(std::string("sample-address"));
    pk.pack(curr_sample.sample_address);

    pk.pack(std::string("sample-instance"));
    pk.pack(curr_sample.sample_instance);

    pk.pack(std::string("proposal-name"));
    pk.pack(std::string(curr_sample.proposal_name));

    pk.pack(std::string("prev-sample-address"));
    pk.pack(prev_sample.prev_sample_address);

    pk.pack(std::string("prev-sample-instance"));
    pk.pack(prev_sample.prev_sample_instance);

    pk.pack(std::string("prev-sample-value"));
    pk.pack(prev_sample.prev_sample_value);
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample_impl(Distr<Params ...>& distr, const bool from_observe) {
    typename Distr<Params ...>::result_type x;
    std::string addr = get_addr();
    auto id = ids.emplace(addr, static_cast<int>(ids.size())).first->second;

    if (id >= static_cast<int>(t.x_.size()))
        t.x_.resize(id + 1);

    if (training) {
        x = distr(rng);

        if(from_observe){
            t.observes_.emplace_back(x);
        }
        else{
            // Lua starts with 1
            t.samples_.emplace_back(Sample{t.time_index_, static_cast<int>(t.x_[id].size()) + 1, x, distr_name<Distr>::value, addr});
        }
    }
    else {
        // Request client
        // TODO(Lezcano) Use proposal distribution to sample given the parameters from the NN
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);

        // Lua starts with 1
        int sample_instance = t.x_[id].size() + 1;
        curr_sample = SampleInference{addr, sample_instance, distr_name<Distr>::value};

        pack_request(pk);

        zmq::message_t request (sbuf.size());
        memcpy (request.data(), sbuf.data(), sbuf.size());
        socket->send(request);

        auto params = receive<std::vector<double>>(*socket);

        // TODO(Lezcano) Use last_x_ and id to compute x
        // x = sample_distr(posterior_distr(params));
        prev_sample = curr_sample;
        prev_sample.prev_sample_value = std::exchange(x, distr(rng));

        // TODO(Lezcano) Accumulate log(p/q) where q is the proposal distribution
    }

    t.x_[id].emplace_back(static_cast<double>(x));

    t.x_addr_.emplace_back(static_cast<double>(x), id);
    ++t.time_index_;

    return x;
}

template<template <class ...> class Distr, class ...Params>
typename Distr<Params ...>::result_type sample(Distr<Params ...>& distr) {
    return sample_impl(distr, false);
}

template<template <class ...> class Distr, class ...Params>
void observe(Distr<Params ...>& distr, double x) {
    if (training){
        sample_impl(distr, true);
    }
    else{
        using std::log;
        auto prob = pdf(math_distr(distr), x);
        t.y_.emplace_back(prob);
        t.log_w_ += log(prob);
    }
}

void send_observe_init(std::vector<double>&& data){
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(&sbuf);

    pk.pack_map(2);
        pk.pack(std::string("command"));
        pk.pack(std::string("observe-init"));

        pk.pack(std::string("command-param"));
        pk.pack_map(2);
            pk.pack(std::string("shape"));
            pk.pack_array(1);
                pk.pack(data.size());

            pk.pack(std::string("data"));
            pk.pack(data);

    zmq::message_t request (sbuf.size()), reply;
    memcpy (request.data(), sbuf.data(), sbuf.size());
    socket->send (request);

    // TODO (Lezcano) This answer is unnecessary
    socket->recv (&reply);
    auto rpl = std::string(static_cast<char*>(reply.data()), reply.size());

    std::string answer = msgpack::unpack(rpl.data(), rpl.size()).get().as<std::string>();
    if(answer != "observe-received")
        std::cout << "Invalid command " << answer << std::endl;
}


template double sample(boost::random::normal_distribution<double>&);

template void observe(boost::random::normal_distribution<double>&, double);
}  // namespace cpprob
