#include <fstream>
#include <string>
#include <cstdlib>

#include "cpprob/state.hpp"
#include "cpprob/serialization.hpp"
#include "models/sherpa.hpp"
#include "models/calorimeter.hpp"

int main(int argc,char* argv[])
{
    using namespace cpprob; // serialization
    State::set(StateType::dryrun);

    models::SherpaWrapper s;

    if (argc < 2) {
        std::cerr << "Please specify the output file and optionally the channel to be sampled.\n";
        std::cerr << "You may optionally specify the number of retries before declaring the sampling unsuccessful. By default this number is 10,000\n";
        std::cerr << "sherpa_gen [output_file] (channel) (retires = 10,000)\n";
    }
    std::string outputfilename = argv[1];
    int channel = -1;
    if (argc < 4) {
        channel = std::stoi(argv[2]);
    }
    int n = 10'000;
    int init_tries = n;
    if (argc < 5) {
        channel = std::stoi(argv[3]);
    }

    std::ofstream file_chan(outputfilename + "_chan.txt");
    std::ofstream file_mom(outputfilename + "_mom.txt");
    std::ofstream file_obs(outputfilename + "_obs.txt");
    decltype(s.sherpa()) tup;
    do {
        n--;
        if (n == -1) break;
        tup = s.sherpa();
    } while (channel != -1 && std::get<0>(tup) != channel);
    if (n == -1) {
        std::cerr << "Could not sample the particle in " << init_tries << "retries.\n"
                  << "Try with a higher number of retries.\n";
        std::exit (EXIT_FAILURE);
    }

    file_chan << std::get<0>(tup) << std::endl;
    file_mom <<  std::get<1>(tup) << std::endl;
    file_obs <<  models::calo_simulation(std::get<2>(tup)) << std::endl;
}
