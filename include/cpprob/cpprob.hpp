#ifndef INCLUDE_CPPROB_HPP
#define INCLUDE_CPPROB_HPP


#include <array>                                        // for array, tuple_...
#include <initializer_list>                             // for initializer_list
#include <iostream>                                     // for operator<<
#include <iterator>                                     // for make_move_ite...
#include <stdexcept>                                    // for runtime_error
#include <string>                                       // for string
#include <tuple>                                        // for tuple
#include <type_traits>                                  // for declval, enab...
#include <utility>                                      // for move, index_s...
#include <vector>                                       // for vector

#include <boost/filesystem/path.hpp>                    // for path

#include "cpprob/call_function.hpp"                     // for call_f_defaul...
#include "cpprob/distributions/utils_distributions.hpp" // for logpdf
#include "cpprob/sample.hpp"                            // for Sample
#include "cpprob/socket.hpp"                            // for SocketInfer
#include "cpprob/state.hpp"                             // for StateInfer
#include "cpprob/traits.hpp"                            // for tuple_size
#include "cpprob/utils.hpp"                             // for get_rng, get_...

namespace cpprob {

template<class Distribution, class String>
auto sample(Distribution && distr, const bool control, String && address)
{
    using distribution_t = std::decay_t<Distribution>;

    if (!control || State::dryrun() || State::sis()) {
        return distr(get_rng());
    }

    typename distribution_t::result_type x{};

    if (State::compile()) {
        x = distr(get_rng());
        StateCompile::add_sample(std::forward<String>(address), std::forward<Distribution>(distr), x);
    }
    else if (State::csis()) {
        StateInfer::new_sample(address, distr);
        double radon_nikodym = 1;

        try {
            auto proposal = StateInfer::get_proposal<proposal_t<distribution_t>>();
            x = proposal(get_rng());
            radon_nikodym = logpdf<distribution_t>()(distr, x) - logpdf<proposal_t<distribution_t>>()(proposal, x);
        }
        // No proposal -> Default to prior as proposal
        catch (const std::runtime_error &) {
            x = distr(get_rng());
            radon_nikodym = 1;
        }
        StateInfer::increment_log_prob(radon_nikodym, address);

        StateInfer::add_value_to_sample(x);
    }
    else {
        std::cerr << "Incorrect branch in sample_impl!!" << std::endl;
    }

    return x;
}

template<class Distribution>
auto sample(Distribution && distr, const bool control=false)
{
    // Repeated code for efficiency
    if (!control || State::dryrun() || State::sis()) {
        return distr(get_rng());
    }
    return sample(distr, control, get_addr());
}


template<class Distribution>
void observe(Distribution && distr, const typename std::decay_t<Distribution>::result_type & x)
{
    using distribution_t = std::decay_t<Distribution>;
    if (State::compile()) {
        StateCompile::add_observe(distr(get_rng()));
    }

    else if (State::csis() || State::sis()) {
        StateInfer::increment_log_prob(logpdf<distribution_t>()(distr, x), "");
    }
}

template<class T,  class String>
void predict(T && x, String && addr)
{
    if (State::csis() || State::sis()) {
        StateInfer::add_predict(std::forward<T>(x), std::forward<String>(addr));
    }
}

template<class T>
void predict(T && x)
{
    if (State::csis() || State::sis()) {
        StateInfer::add_predict(std::forward<T>(x), get_addr());
    }
}

template<class T>
void metaobserve(T && x)
{
    if (State::compile()) {
        StateCompile::add_observe(std::forward<T>(x));
    }
}

struct rejection_sampling {
    rejection_sampling();
    rejection_sampling& operator=(const rejection_sampling&) = delete;
    rejection_sampling& operator=(rejection_sampling &&) = delete;
    ~rejection_sampling();
};

void start_rejection_sampling();

void finish_rejection_sampling();

template<class Func>
void compile(const Func & f,
             const std::string & tcp_addr,
             const std::string & dump_folder,
             int batch_size,
             int n_batches=0)
{
    const bool online_training = dump_folder.empty();
    State::set(StateType::compile);

    if (online_training) {
        SocketCompile::connect_server(tcp_addr);
    }
    else {
        SocketCompile::config_file(dump_folder);
    }

    // It will break if n_batches != 0 or if it gets a RequestFinishCompilation
    for (std::size_t i = 0; /* Forever */ ; ++i) {
        std::cout << "Generating batch " << i << std::endl;
        StateCompile::start_batch();
        if (online_training) {
            // TODO(Lezcano) C++17 Change to std::optional<int>, right now this is a hacky hack
            batch_size = SocketCompile::get_batch_size();
            if (batch_size == -1) {
                SocketCompile::send_finish_compilation();
                break;
            }
        }

        for (int n = 0; n < batch_size; ++n) {
            StateCompile::start_trace();
            call_f_default_params(f);
            StateCompile::finish_trace();
        }
        StateCompile::finish_batch();
        // If n_batches != 0, keep track of the generated traces
        if (n_batches != 0) {
            --n_batches;
            if (n_batches < 0) {
                break;
            }
        }
    }
}

template<class Func, class... Args>
void inference(
        const StateType algorithm,
        const Func & f,
        const std::tuple<Args...> & observes,
        std::size_t n = 50'000,
        const boost::filesystem::path & file_name = "posterior",
        const std::string & tcp_addr = "")
{
    static_assert(sizeof...(Args) != 0, "The function has to receive the observed values as parameters.");

    State::set(algorithm);
    StateInfer::start_infer();

    if (State::csis()) {
        SocketInfer::connect_client(tcp_addr);
        StateInfer::send_start_inference(observes);
    }

    StateInfer::config_file(file_name);

    for (std::size_t i = 0; i < n; ++i) {
        if (i % 100 == 0) {
            std::cout << "Generating trace " << i << std::endl;
        }
        StateInfer::start_trace();
        call_f_tuple(f, observes);
        StateInfer::finish_trace();
    }
    StateInfer::finish_infer();
}

} // end namespace cpprob
#endif //INCLUDE_CPPROB_HPP
