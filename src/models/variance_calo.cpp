#include <fstream>
#include <string>
#include <vector>
#include "cpprob/serialization.hpp"
#include "cpprob/calorimeter.hpp"

void from_file(const std::string & file_name)
{
    using namespace cpprob;

    std::vector<std::vector<double>> final_state_particles;
    std::ifstream f{file_name};
    f >> final_state_particles;

    calo_histo = calo_simulation(final_state_particles);
    std::cout << calo_histo << std::endl;
}

void to_file(const std::string & file_name)
{
    using namespace cpprob;

    models::SherpaWrapper s;
    std::ofstream f{file_name};
    auto tup = s.sherpa();
    f << std::get<2>(tup);
}
