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

#include "cpprob/call_function.hpp"                     // for call_f_defaul...
#include "cpprob/distributions/utils_distributions.hpp" // for logpdf
#include "cpprob/metapriors.hpp"                        // for discard_build
#include "cpprob/sample.hpp"                            // for Sample
#include "cpprob/socket.hpp"                            // for SocketInfer
#include "cpprob/state.hpp"                             // for StateInfer
#include "cpprob/traits.hpp"                            // for tuple_size
#include "cpprob/utils.hpp"                             // for get_rng, get_...

namespace cpprob {

// Declared at the top of state.hpp with control=false
template<class Distribution>
typename Distribution::result_type sample(Distribution & distr, const bool control)
{

    if (!control ||
        State::state() == StateType::dryrun ||
        State::state() == StateType::importance_sampling) {
        return distr(get_rng());
    }

    typename Distribution::result_type x{};
    std::string addr {get_addr()};

    if (State::state() == StateType::compile) {
        x = distr(get_rng());
        StateCompile::add_sample(addr, distr, x);
    }
    else if (State::state() == StateType::inference) {
        StateInfer::new_sample(addr, distr);

        try {
            auto proposal = StateInfer::get_proposal<Distribution>();
            x = proposal(get_rng());

            StateInfer::increment_log_prob(logpdf<Distribution>()(distr, x) - logpdf<proposal_t<Distribution>>()(proposal, x));
        }
        // No proposal -> Default to prior as proposal
        catch (const std::runtime_error &) {
            x = distr(get_rng());
        }

        StateInfer::add_value_to_sample(x);
    }
    else {
        std::cerr << "Incorrect branch in sample_impl!!" << std::endl;
    }

    return x;
}


template<class Distribution>
void observe(Distribution & distr, const typename Distribution::result_type & x)
{
    if (State::compile()) {
        StateCompile::add_observe(distr(get_rng()));
    }
    else if (State::inference()) {
        StateInfer::increment_log_prob(logpdf<Distribution>()(distr, x));
    }
}

// Declared at the top of state.hpp with addr = ""
template<class T>
void predict(const T & x, const std::string & addr)
{
    if (State::inference()) {
        if (addr.empty()) {
            StateInfer::add_predict(x, get_addr());
        }
        else {
            StateInfer::add_predict(x, addr);
        }
    }
}

void start_rejection_sampling();

void finish_rejection_sampling();

template<class Func>
void compile(const Func & f,
             const std::string & tcp_addr,
             const std::string & dump_folder,
             std::size_t batch_size)
{
    State::set(StateType::compile);
    const bool to_file = !dump_folder.empty();

    if (to_file) {
        SocketCompile::config_file(dump_folder);
    }
    else {
        SocketCompile::connect_server(tcp_addr);
    }

    for (std::size_t i = 0; /* Forever */ ; ++i) {
        std::cout << "Generating batch " << i << std::endl;
        StateCompile::start_batch();
        if (!to_file) {
            batch_size = SocketCompile::get_batch_size();
        }

        for (std::size_t i = 0; i < batch_size; ++i) {
            StateCompile::start_trace();
            call_f_default_params(f);
            StateCompile::finish_trace();
        }
        StateCompile::finish_batch();
    }
}

template<class Func, class... Args>
void generate_posterior(
        const Func & f,
        const std::tuple<Args...> & observes,
        const std::string & tcp_addr,
        const std::string & file_name,
        std::size_t n,
        const StateType state)
{
    static_assert(sizeof...(Args) != 0, "The function has to receive the observed values as parameters.");

    State::set(state);
    StateInfer::start_infer();

    if (State::state() == StateType::inference) {
        SocketInfer::connect_client(tcp_addr);
        StateInfer::send_start_inference(observes);
    }

    SocketInfer::config_file(file_name);

    for (std::size_t i = 0; i < n; ++i) {
        std::cout << "Generating trace " << i << std::endl;
        StateInfer::start_trace();
        call_f_tuple(f, observes);
        StateInfer::finish_trace();
    }
    StateInfer::finish_infer();
}

} // end namespace cpprob
#endif //INCLUDE_CPPROB_HPP
