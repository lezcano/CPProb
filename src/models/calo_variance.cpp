#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cpprob/serialization.hpp"
#include "models/calorimeter.hpp"
#include "models/sherpa.hpp"

std::vector<std::vector<std::vector<double>>> generate_response_from_file(const std::string & file_name)
{
    using namespace cpprob;

    int chan;
    std::vector<double> mother_momentum;
    std::vector<std::vector<double>> final_state_particles;
    std::ifstream f{file_name};
    f >> chan;
    f >> mother_momentum;
    f >> final_state_particles;

    auto calo_histo = calo_simulation(final_state_particles);
    return calo_histo;
}


void to_file(const std::string & file_name, int seed)
{
    using namespace cpprob;

    models::SherpaWrapper s;
    std::ofstream f{file_name};


    auto old_rng = get_rng();
    get_rng().seed(seed);
    auto tup = s.sherpa();
    get_rng() = old_rng;

    f << std::get<0>(tup) << '\n';
    f << std::get<1>(tup) << '\n';
    f << std::get<2>(tup) << '\n';
}

void save_avg_var_data(const std::string & file_name)
{
    using namespace cpprob;

    models::SherpaWrapper s;
    std::ofstream f{file_name};
    auto tup = s.sherpa();
    f << std::get<0>(tup);
    f << "\n";
    f << std::get<1>(tup);
    f << "\n";
    f << std::get<2>(tup);
    f << "\n";
}

double average(const std::vector<double> & vec){
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double variance(const std::vector<double> & vec, double avg){
    double sum_devsq = std::accumulate(vec.begin(), vec.end(), 0.0,
                                       [avg](double acc, double value) { return acc + ((value - avg) * (value - avg)); });
    return sum_devsq / vec.size();
}

int main(int argc, char* argv[]){

    if (argc < 2) {
        std::cout << "need mode" << std::endl;
        return 1;
    }

    int mode = atoi(argv[1]);

    std::cout << "the variance: mode: " << mode << std::endl;

    if (mode == 1) {
        int seed = atoi(argv[3]);
        to_file(argv[2], seed);
    }

    auto voxel_value_lists =
        std::vector<std::vector<std::vector<std::vector<double>>>>(35,
            std::vector<std::vector<std::vector<double>>>(35,
              std::vector<std::vector<double>>(20,std::vector<double>())
            )
        );

    if (mode == 2) {
        std::size_t n_iterations = atoi(argv[5]);

        for(std::size_t iter = 0; iter<n_iterations;++iter){
            if (iter % 100 == 0) { std::cout << "iter: " << iter << std::endl; }
            auto calo_histo = generate_response_from_file(argv[2]);
            for (std::size_t ix = 0; ix < calo_histo.size(); ++ix) {
                for (std::size_t iy = 0; iy < calo_histo[ix].size(); ++iy) {
                    for (std::size_t iz = 0; iz < calo_histo[ix][iy].size(); ++iz) {

                        // std::cout << "val: " << calo_histo[ix][iy][iz] << std::endl;
                        voxel_value_lists[ix][iy][iz].push_back(calo_histo[ix][iy][iz]);
                        // std::cout << "voxl: " << voxel_value_lists[ix][iy][iz] << std::endl;
                    }
                }
            }
        }



        auto avg =
            std::vector<std::vector<std::vector<double>>>(35,
                std::vector<std::vector<double>>(35,
                    std::vector<double>(20)
                 )
            );
        auto var =
            std::vector<std::vector<std::vector<double>>>(35,
                std::vector<std::vector<double>>(35,
                    std::vector<double>(20)
                )
            );


        for (std::size_t ix = 0; ix < voxel_value_lists.size(); ++ix){
            for (std::size_t iy = 0; iy < voxel_value_lists[ix].size(); ++iy){
                for (std::size_t iz = 0; iz < voxel_value_lists[ix][iy].size(); ++iz){
                    double avg_val = average(voxel_value_lists[ix][iy][iz]);
                    double var_val  = variance(voxel_value_lists[ix][iy][iz], avg_val);
                    // std::cout << "vox: " << voxel_value_lists[ix][iy][iz] << " var: " << var_val << " avg: " << avg_val << std::endl;
                    avg[ix][iy][iz] = avg_val;
                    var[ix][iy][iz] = var_val;
                }
            }
        }

        std::ofstream avg_f(argv[3]);
        avg_f << avg;

        std::ofstream var_f(argv[4]);
        var_f << var;

    }


    return 0;
}
