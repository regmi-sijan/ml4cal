#include <onnxruntime_cxx_api.h>

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

#include "lyra.hpp"
#include "onnxutil.h"
#include "onnxlib.h"

using namespace std::chrono;



int main(int argc, char* argv[]) {

    bool help               = false;
    bool verbose            = false;

    int N                   = 0;

    std::string modelfile   = "tfmodel.onnx";
    std::string rootfile    = "rootfile.root";


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
        | lyra::opt(N, "N")
            ["-N"]["--Nentries"]
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
        std::cout << "*** Verbose mode selected" << std::endl << "*** Model file: " << modelfile << std::endl;
        std::cout << "*** ROOT file: " << rootfile << std::endl;
    }
 
    std::string instanceName{"fit"};

    OnnxSession* oS = new OnnxSession(modelfile.c_str(), instanceName.c_str());

    size_t numInputNodes = oS->_numInputNodes, numOutputNodes = oS->_numOutputNodes;
    std::vector<int64_t> inputDims = oS->_inputDimensions, outputDims = oS->_outputDimensions;
    const char* inputName = oS->_inputName; const char* outputName = oS->_outputName;
    ONNXTensorElementDataType inputType = oS->_inputType, outputType = oS->_outputType;

    if(verbose) {
        std::cout << "*** Input Nodes:\t" << numInputNodes << ",\t\t Output Nodes: " << numOutputNodes << std::endl;
        std::cout << "*** Input Name:\t\t"            << inputName     << ",\t Input Type: "  << inputType  << ",\t Input Dimensions:\t" << inputDims << std::endl;
        std::cout << "*** Output Name:\t"           << outputName    << ",\t Output Type: " << outputType << ",\t Output Dimensions:\t"<< outputDims<< std::endl;
    }


    size_t inputTensorSize  = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);

    //Declare container for input data:
    std::vector<float> inputTensorValues(inputTensorSize);

    // Proceed to read the data from a ROOT tree:
    TFile f(TString(rootfile.c_str()));
    if (f.IsZombie()) { cout << "Error opening file" << endl; exit(-1);}
    if (verbose) { cout << "*** Input ROOT file " << rootfile << " has been opened" << endl;}

    TTree   *tree       = (TTree*)f.Get("trainingtree;1");
    TBranch *branch     = tree->GetBranch("waveform");

    Int_t waveform[64][32];
    branch->SetAddress(&waveform);      // will read into this array
    Long64_t n = branch->GetEntries();  // number of entries in the branch
    if( N==0 || N>n ) { N=n; }          // decide how many to process, 0=all

    if(verbose) { std::cout << "*** Number of entries in the file: " << n << ", Number of entries to be processed: " << N << std::endl; }

    std::vector<float>          outputTensorValues(outputTensorSize);

    std::vector<const char*>    inputNames{inputName};
    std::vector<const char*>    outputNames{outputName};
    std::vector<Ort::Value>     inputTensors;
    std::vector<Ort::Value>     outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (int i=0; i<N; i++) {
        Int_t m = branch->GetEntry(i);


        // std::transform(intVec.begin(), intVec.end(), doubleVec.begin(), [](int x) { return (double)x;});
        // inputTensorValues = std::vector<int>({1,2})

        for(int bin=0; bin<31; bin++) {
            inputTensorValues[bin] = (float) waveform[27][bin];
        }

        // cout << inputTensorValues << endl;
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size()));

        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));
    
        // Get starting timepoint
        auto start = chrono::high_resolution_clock::now();

        oS->_session->Run(Ort::RunOptions{nullptr}, inputNames.data(),        
            inputTensors.data(), 1, outputNames.data(),
            outputTensors.data(), 1);  
        
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<microseconds>(stop - start);
        cout << "Microseconds: " << duration.count() << endl;
        
        std::cout << outputTensorValues << std::endl;

        // vector<float> data(a,a + sizeof( a ) / sizeof( a[0] ) );
        // inputTensorValues = waveform[27];
    }
    exit(0);
}
        // if(verbose) {for(int bin=0; bin<32; bin++) {cout<< waveform[27][bin] << " ";} cout << endl; }
        // cout << sizeof(waveform[27]) <<endl;
        // vector<float> a[31];