#include <fstream>
#include <string>
#include <cstdlib>

#include "cpprob/cpprob.hpp"
#include "cpprob/model.hpp"
#include "cpprob/ndarray.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/state.hpp"
#include "models/sherpa.hpp"

int main(int argc,char* argv[])
{
    using namespace cpprob::detail;
    cpprob::State::set(cpprob::StateType::dryrun);

    cpprob::models::SherpaWrapper s;

    if (argc != 4 && argc != 5) {
        std::cerr << "Please specify 3 or 4 arguments.\n";
        std::cerr << "sherpa_gen [output_file] [number_observes] [number_particles] [NN_addr](=tcp://127.0.0.1:6666)\n";
        std::exit (EXIT_FAILURE);
    }

    std::string outputfilename = argv[1];
    int n_observes = std::stoi(argv[2]);
    int n_particles = std::stoi(argv[3]);

    std::string tcp_addr;
    if (argc == 5) {
        tcp_addr = argv[4];
    }
    else {
        tcp_addr = "tcp://127.0.0.1:6666";
    }


    std::ofstream file_chan(outputfilename + "_chan.txt");
    std::ofstream file_chan_pred (outputfilename + "_chan_pred.txt");
    std::ofstream file_chan_distr (outputfilename + "_chan_distr.txt");

    std::ofstream file_mom(outputfilename + "_mom.txt");
    std::ofstream file_mom_pred (outputfilename + "_mom_pred.txt");
    std::ofstream file_mom_distr (outputfilename + "_mom_distr.txt");

    std::ofstream file_obs(outputfilename + "_obs.txt");

    cpprob::Model<> model;
    std::array<std::array<double, 38>, 38> conf_matrix{};
    for (int i = 0; i < n_observes; ++i) {
        auto tup = s.sherpa_pred_obs();
        file_chan << std::get<0>(tup) << std::endl;
        file_mom <<  std::get<1>(tup) << std::endl;
        file_obs <<  std::get<2>(tup) << std::endl;

        cpprob::generate_posterior(s, make_tuple(std::get<2>(tup)), tcp_addr, "tmp.obs", n_particles);
        std::ifstream tmp ("tmp.obs");
        model.load_points(tmp);

        auto distr_map_channel = model.distribution_map<int>(0, 0);
        file_chan_distr << distr_map_channel.first << std::endl;
        file_chan_pred << distr_map_channel.second.values() << std::endl;

        conf_matrix[std::get<0>(tup)][static_cast<int>(distr_map_channel.second)]++;

        auto mean_variance_mom = model.mean_variance(1, 0); // Momentum
        file_chan_distr << model.distribution<double>(1, 0) << std::endl;
        file_chan_pred << mean_variance_mom.first  << " " << mean_variance_mom.second << std::endl;

    }
    for (auto & row : conf_matrix) {
        auto sum = std::accumulate(row.begin(), row.end(), 0.0);
        for (auto & elem : row) {
            elem /= sum;
        }
    }
    std::ofstream file_conf_matrix(outputfilename + "_conf_matrix.txt");
    file_conf_matrix << conf_matrix << std::endl;
}
