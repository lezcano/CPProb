#include <fstream>
#include <string>

#include "cpprob/serialization.hpp"
#include "models/sherpa.hpp"

int main(int argc,char* argv[])
{
    using namespace cpprob::detail;

    sherpa_detail::SherpaWrapper s;

    std::string outputfilename = "out.txt";
    if( argc > 1){
        outputfilename = argv[1];
    }
    std::ofstream file (outputfilename);
    file << s.sherpa() << std::endl;
}
