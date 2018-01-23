#ifndef CPPROB_STATS_PRINTER_HPP
#define CPPROB_STATS_PRINTER_HPP

#include <cstdlib>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

#include <boost/filesystem/operations.hpp>                  // for exists
#include <boost/filesystem/path.hpp>                        // for path

#include "cpprob/ndarray.hpp"
#include "cpprob/state.hpp"
#include "cpprob/serialization.hpp"
#include "cpprob/postprocess/empirical_distribution.hpp"

namespace cpprob {
// TODO(Lezcano) Maybe factor into two classes
class StatsPrinter {
public:

    StatsPrinter(const std::string & file_path)
    {
        namespace bf = boost::filesystem;
        auto file_ids = file_path + ".ids";
        if (!bf::exists(bf::path(file_ids))) {
            return;
        }
        std::ifstream ids_file(file_ids.c_str());
        for (std::string line; std::getline(ids_file, line);) {
            ids_.emplace_back(std::move(line));
        }

        load_distr(file_path + ".int", int_distr_);
        load_distr(file_path + ".real", real_distr_);
    }

    friend std::ostream & operator<<(std::ostream & out, const StatsPrinter & sp)
    {
        out << "Estimators for " << sp.file_name_ << std::endl;
        for (const auto & kv : sp.real_distr_) {
            std::size_t i = 0;
            for (const auto & emp_distr : kv.second) {
                out << sp.ids_[kv.first];
                if (kv.second.size() > 1) {
                    out << ' ' << i;
                }
                out << ':' << std::endl;
                auto mean = emp_distr.mean();
                out << "  Mean: " << mean << std::endl
                    << "  Variance: " << emp_distr.variance(mean) << std::endl;
                ++i;
            }
        }
        for (const auto & kv : sp.int_distr_) {
            std::size_t i = 0;
            for (const auto & emp_distr : kv.second) {
                out << sp.ids_[kv.first];
                if (kv.second.size() > 1) {
                    out << ' ' << i;
                }
                out << ':' << std::endl
                    << "  Distribution:\n";
                auto distr = emp_distr.distribution();
                for (const auto & x_w : distr) {
                    out << "    " << x_w.first << ": " << x_w.second << std::endl;
                }
                out << "  MAP: " << emp_distr.max_a_posteriori(distr) << std::endl;
                out << "  Num points: " << emp_distr.num_points() << std::endl;
                ++i;
            }
        }
        return out;
    }

private:
    // Attributes
    std::map<std::size_t, std::vector<EmpiricalDistribution<int>>> int_distr_;
    std::map<std::size_t, std::vector<EmpiricalDistribution<NDArray<double>>>> real_distr_;
    std::vector<std::string> ids_;
    std::string file_name_;

    template<class T>
    void load_distr(const std::string & file_name, std::map<std::size_t, std::vector<EmpiricalDistribution<T>>> & distributions)
    {
        std::ifstream file(file_name.c_str());

        if (!file.is_open()) {
            return;
        }

        for (std::string line; std::getline(file, line);) {
            std::map<std::size_t, std::size_t> predict_instance;
            std::pair<std::vector<std::pair<std::size_t, T>>, double> predicts;
            std::istringstream iss(line);
            if (!(iss >> predicts)) {
                std::cerr << "Bad format in line:\n" << line << std::endl;
                std::exit(EXIT_FAILURE);
            }

            for (const auto & elem : predicts.first) {
                auto& vec_distr = distributions[elem.first];
                auto& num_times_hit_predict = predict_instance[elem.first];
                // Create new distribution for that predict statement if the vector of distributions is not long enough
                if (num_times_hit_predict == vec_distr.size()) {
                    vec_distr.emplace_back();
                    vec_distr.back().add_point(elem.second, predicts.second);
                }
                else {
                    vec_distr[num_times_hit_predict].add_point(elem.second, predicts.second);
                }
                ++num_times_hit_predict;
            }
        }
    }
};

} // end namespace cpprob
#endif //CPPROB_STATS_PRINTER_HPP
