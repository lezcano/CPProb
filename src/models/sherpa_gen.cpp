#include <fstream>
#include <string>
#include <cstdlib>

#include "cpprob/state.hpp"
#include "cpprob/serialization.hpp"
#include "models/sherpa.hpp"

int main(int argc,char* argv[])
{
    using namespace cpprob; // serialization
    State::set(StateType::dryrun);

    models::SherpaWrapper s;

    if (argc != 3) {
        std::cerr << "Please specify the output file and the number of particles\n";
        std::cerr << "sherpa_gen [output_file] [number_particles]\n";
        std::exit (EXIT_FAILURE);
    }
    std::string outputfilename = argv[1];
    int n = std::stoi(argv[2]);

    std::ofstream file_chan(outputfilename + "_chan.txt");
    std::ofstream file_mom(outputfilename + "_mom.txt");
    std::ofstream file_obs(outputfilename + "_obs.txt");
    for (int i = 0; i < n; ++i) {
        auto tup = s.sherpa();
        file_chan << std::get<0>(tup) << std::endl;
        file_mom <<  std::get<1>(tup) << std::endl;
        file_obs <<  std::get<2>(tup) << std::endl;
    }
}
