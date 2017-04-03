
class Trace{
public:

    Core(bool training, zmq::socket_t* socket = nullptr);

private:

    template<class Func>
    friend std::vector<std::vector<double>>
    expectation(const Func&,
                std::vector<double>,
                size_t,
                const std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)>&);

    template<typename Func>
    friend Core eval(Func f, bool training, zmq::socket_t* socket);

    int time_index_ = 1;
    double w_ = 0;
    std::vector<std::vector<double>> x_;
    static std::unordered_map<std::string, int> ids_;

    std::vector<std::pair<double, int>> x_addr_;
    std::vector<double> y_;

    std::vector<Sample> samples_;
    std::vector<double> observes_;

    bool training_;
    zmq::socket_t* socket_ = nullptr;

    int prev_sample_instance_ = 0;
    double prev_x_ = 0;
    std::string prev_addr_ = "";
};


class PrevSampleInference{
public:
    SampleInferencePrev(const SampleInference& s) :
        prev_sample_address{s.sample_address},
        prev_sample_instance{s.sample_instance}; {}

    SampleInferencePrev& operator=(const SampleInference& s){
        prev_sample_address = s.sample_address;
        prev_sample_instance = s.sample_instance;
    }
private:
    std::string prev_sample_address;
    int prev_sample_instance;
    double prev_value;
}

class SampleInference{
public:

private:
    std::string sample_address;
    int sample_instance;
    std::string proposal_name;
    friend class SampleInferencePrev;
}
