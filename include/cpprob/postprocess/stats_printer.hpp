#ifndef CPPROB_STATS_PRINTER_HPP
#define CPPROB_STATS_PRINTER_HPP

#include <cstdlib>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include "cpprob/ndarray.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/postprocess/empirical_distribution.hpp"

namespace cpprob {
class Printer {
public:

    void load(const std::string & file_name)
    {
        std::ifstream ids_file(file_name + "_ids");
        for (std::string line; std::getline(ids_file, line);) {
            ids_.emplace_back(std::move(line));
        }

        load_distr(file_name + "_int", int_distr_);
        load_distr(file_name + "_real", real_distr_);
    }

    void print(std::ostream & out) const
    {
        for (const auto & kv : real_distr_) {
            for (std::size_t i = 0; i < kv.second.size(); ++i) {
                auto mean = kv.second[i].mean();
                out << ids_[kv.first] << " " << i << ":" << std::endl
                    << "  Mean: " << mean << std::endl
                    << "  Variance: " << kv.second[i].variance(mean) << std::endl;
            }
        }
        for (const auto & kv : int_distr_) {
            for (std::size_t i = 0; i < kv.second.size(); ++i) {
                out << ids_[kv.first] << " " << i << ":" << std::endl
                    << "  Distribution:\n";
                auto distr = kv.second[i].distribution();
                for (const auto & x_w : distr) {
                    out << "    " << x_w.first << ": " << x_w.second << std::endl;
                }
                out << "  MAP: " << kv.second[i].max_a_posteriori(distr) << std::endl;
            }
        }
    }

private:
    // Attributes
    std::map<int, std::vector<EmpiricalDistribution<int>>> int_distr_;
    std::map<int, std::vector<EmpiricalDistribution<NDArray<double>>>> real_distr_;
    std::vector<std::string> ids_;

    template<class T>
    void load_distr(const std::string & file_name, std::map<int, std::vector<EmpiricalDistribution<T>>> & distributions)
    {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            return;
        }

        int num_points = 0;
        for (std::string line; std::getline(file, line);) {
            std::map<int, int> predict_instance;
            std::pair<std::vector<std::pair<int, T>>, double> predicts;
            std::istringstream iss(line);
            if (!(iss >> predicts)) {
                std::cerr << "Bad format in line:\n" << line << std::endl;
                std::exit(EXIT_FAILURE);
            }

            for (const auto & elem : predicts.first) {
                auto& vec_distr = distributions[elem.first];
                int& num_times_hit_predict = predict_instance[elem.first];
                // Create new distribution for that predict statement if the vector of distributions is not long enough
                if (num_times_hit_predict == static_cast<int>(vec_distr.size())) {
                    vec_distr.emplace_back();
                    vec_distr.back().add_point(elem.second, predicts.second);
                }
                else {
                    vec_distr[num_times_hit_predict].add_point(elem.second, predicts.second);
                }
                ++num_times_hit_predict;
            }
            ++num_points;
        }
        for (auto & addr_distrs : distributions) {
            for (auto & distr : addr_distrs.second) {
                distr.set_num_points(num_points);
            }
        }
    }
};

} // end namespace cpprob
#endif //CPPROB_STATS_PRINTER_HPP
