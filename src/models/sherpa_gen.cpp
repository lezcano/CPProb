#include <ofstream>
#include <string>

#include "cpprob/serialization.hpp"

int main(int argc,char* argv[])
{
    using cpprob::detail;

    std::string outputfilename = "out.txt";
    if( argc > 1){
        outputfilename = argv[1];
    }
    std::ofstream << sherpa_wrapper() << std::endl;
}
