#include <fstream>
#include <string>

#include "cpprob/state.hpp"
#include "cpprob/serialization.hpp"
#include "models/sherpa.hpp"

int main(int argc,char* argv[])
{
    using namespace cpprob::detail;
    cpprob::State::set(cpprob::StateType::dryrun);

    cpprob::models::SherpaWrapper s;

    std::string outputfilename = "out.txt";
    if( argc > 1){
        outputfilename = argv[1];
    }
    std::ofstream file (outputfilename);
    file << s.sherpa() << std::endl;
}
