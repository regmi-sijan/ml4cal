#include <iostream>

#include "lyra.hpp"
#include "TFile.h"

using namespace std;

int main(int argc, char* argv[]) {

    bool help               =   false;
    bool verbose            =   false;
    std::string modelfile   =   "tfmodel.onnx";
    std::string rootfile    =   "";


    auto cli = lyra::cli()
        | lyra::opt(verbose)
            ["-v"]["--verbose"]
            ("verbose" )
        | lyra::opt(modelfile, "model" )
            ["-m"]["--model"]
            ("model")
        | lyra::opt(rootfile, "root" )
            ["-r"]["--root"]
            ("root")            
        | lyra::help(help);

    auto result = cli.parse( { argc, argv } );
    if ( !result ) {
            std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
            exit(1);
    }


    if(help) {
        std::cout << cli << std::endl;
        exit(0);
    }

    if(verbose) {
        std::cout << "Verbose mode selected" << std::endl;
        std::cout << "Model file: " << modelfile << std::endl;
    }

    TFile f(TString(rootfile.c_str()));

    if (f.IsZombie()) {
        cout << "Error opening file" << endl;
        exit(-1);
    }
}
