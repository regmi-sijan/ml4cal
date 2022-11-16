using namespace std;

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <string>

// ROOT
#include "TFile.h"
#include "TTree.h"

// CLI parser
#include "lyra.hpp"

// Wrapper for the ONNX inference:
#include "onnxlib.h"
// Misc utils for general ONNX setup, mostul unused here
#include "onnxutil.h"


using namespace std::chrono;

int main(int argc, char* argv[]) {

    bool help               = false;
    bool verbose            = false;
    bool output             = false;
    bool inspect            = false;

    int N                   = 0;
    int channel             = 27;

    std::string modelfile   = "tfmodel.onnx";
    std::string rootfile    = "rootfile.root";

    auto cli = lyra::cli()
        | lyra::opt(verbose)                ["-v"]["--verbose"] ("Verbose mode")
        | lyra::opt(output)                 ["-o"]["--output"]  ("Print inference output")
        | lyra::opt(inspect)                ["-i"]["--inspect"] ("Inspect")        
        | lyra::opt(modelfile, "model" )    ["-m"]["--model"]   ("File containing the ONNX model")
        | lyra::opt(rootfile, "root" )      ["-r"]["--root"]    ("ROOT file to be read")
        | lyra::opt(N, "N")                 ["-N"]["--Nentries"]("Number of entries to process")                        
        | lyra::opt(channel, "channel")     ["-c"]["--channel"] ("Caloriemter channel")
        | lyra::help(help);

    auto cli_parse = cli.parse( { argc, argv } );
    if( !cli_parse )    {std::cerr << "Error in command line: " << cli_parse.errorMessage() << std::endl; exit(1);}
    if(help)            {std::cout << cli << std::endl; exit(0);}

    if(verbose) {std::cout << "*** Verbose mode selected" << std::endl << "*** Model file: " << modelfile << std::endl << "*** ROOT file: " << rootfile << std::endl;}
 
    std::string instanceName{"fit"};

    std::vector<float> w31(31); // 31 bins in the test beam data; was inputTensorSize
    std::vector<float> input{};

    // Proceed to read the data from a ROOT tree:
    TFile f(TString(rootfile.c_str()));
    if (f.IsZombie()) { cout << "Error opening ROOT file, exiting..." << endl; exit(-1);}
    if (verbose) { cout << "*** Input ROOT file " << rootfile << " has been opened" << endl;}

    TTree   *tree       = (TTree*)f.Get("trainingtree;1");
    TBranch *branch     = tree->GetBranch("waveform");

    Int_t waveform[64][32];

    branch->SetAddress(&waveform);      // will read into this array

    Long64_t n = branch->GetEntries();  // number of entries in the branch

    if( N==0 || N>n ) { N=n; }          // decide how many to process, 0=all
    if(verbose) { std::cout << "*** Number of entries in the file: " << n << ", Number of entries to be processed: " << N << std::endl; }

    for (int i=0; i<N; i++) {
    
        Int_t m = branch->GetEntry(i);
        std::vector<int> inp;
        inp.insert(inp.begin(), std::begin(waveform[channel]), std::end(waveform[channel]));

        if(inspect) {cout << inp <<endl;}

        // in the following line we truncate the 32th element of the array
        std::transform(inp.begin(), inp.end()-1, w31.begin(), [](int x) {return ((float)x)/1000.0;});
        input.insert(input.end(), w31.begin(), w31.end());
    }

    if(verbose) {std::cout << "Input size: " << input.size() << std::endl;}

    // Core inference
    Ort::Session* s = onnxSession(modelfile);
    //+ std::vector<float> result = onnxInference(s, input, N);


    if(output) {
        for (int i=0; i<N*3; i=i+3) {
            //+ for (int k=0; k<3; k++) {std::cout << result[i+k] << " ";}
            std::cout << std::endl;
        }
    }

    //+ delete s;

    exit(0);
}
    
