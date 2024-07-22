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
#include "onnxsession.h"

using namespace std::chrono;


// Based on the inference example by Lei Mao
// https://github.com/leimao/ONNX-Runtime-Inference/blob/main/src/inference.cpp


int main(int argc, char* argv[]) {

    bool help               = false;
    bool verbose            = false;

    int N                   = 0;
    int channel             = 27;

    std::string modelfile   = "tfmodel.onnx";
    std::string rootfile    = "rootfile.root";

    auto cli = lyra::cli()
        | lyra::opt(verbose)
            ["-v"]["--verbose"]
            ("Verbose mode")
        | lyra::opt(modelfile, "model" )
            ["-m"]["--model"]
            ("File containing the ONNX model")
        | lyra::opt(rootfile, "root" )
            ["-r"]["--root"]
            ("ROOT file to be read")
        | lyra::opt(N, "N")
            ["-N"]["--Nentries"]
            ("Number of entries to process")                        
        | lyra::opt(channel, "channel")
            ["-c"]["--channel"]
            ("Calorimeter channel")
        | lyra::help(help);

    auto result = cli.parse( { argc, argv } );
    if( !result ) {std::cerr << "Error in command line: " << result.errorMessage() << std::endl;exit(1);}
    if(help) {std::cout << cli << std::endl; exit(0);}

    if(verbose) {
        std::cout << "*** Verbose mode selected" << std::endl << "*** Model file: " << modelfile << std::endl;
        std::cout << "*** ROOT file: " << rootfile << std::endl;
    }
 
    std::string instanceName{"fit"};

    OnnxSession* oS = new OnnxSession(modelfile.c_str(), instanceName.c_str(), 1); // oS=="ONNX Session"


    std::vector<int64_t> inputDims = oS->inputDimensions(), outputDims = oS->outputDimensions();

    if(verbose) {
        std::cout << "*** Input Nodes:\t"   << oS->numInputNodes()  << ",\t\t Output Nodes: "  << oS->numOutputNodes() << std::endl;
        std::cout << "*** Input Name:\t\t"  << oS->inputName()      << ",\t Input Type: "      << oS->inputType()  << ",\t Input Dimensions:\t"  << inputDims    << std::endl;
        std::cout << "*** Output Name:\t"   << oS->outputName()     << ",\t Output Type: "     << oS->outputType() << ",\t Output Dimensions:\t" << outputDims   << std::endl;
    }


    size_t inputTensorSize = vectorProduct(inputDims), outputTensorSize = vectorProduct(outputDims);

    // Proceed to read the data from a ROOT tree:inputten
    TFile f(TString(rootfile.c_str()));
    if (f.IsZombie()) { cout << "Error opening file" << endl; exit(-1);}
    if (verbose) { cout << "*** Input ROOT file " << rootfile << " has been opened" << endl;}

    TTree   *tree       = (TTree*)f.Get("trainingtree;1");
    TBranch *branch     = tree->GetBranch("waveform");

    Int_t waveform[64][32];
    branch->SetAddress(&waveform);      // will read into this array
    Long64_t n = branch->GetEntries();  // number of entries in the branch
    if( N==0 || N>n ) { N=n;}          // decide how many to process, 0=all

    if(verbose) { std::cout << "*** Number of entries in the file: " << n << ", Number of entries to be processed: " << N << std::endl; }

    std::vector<float>          outputTensorValues(outputTensorSize);

    std::vector<const char*>    inputNames{oS->inputName()}, outputNames{oS->outputName()};
    std::vector<Ort::Value>     inputTensors, outputTensors;

    std::vector<float> w31(inputTensorSize); // 31 bins in the test beam data

    // Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (int i=0; i<N; i++) {
//        Int_t m = branch->GetEntry(i);
        auto start = chrono::high_resolution_clock::now();

        std::vector<int> inp;
        inp.insert(inp.begin(), std::begin(waveform[channel]), std::end(waveform[channel]));

        std::transform(inp.begin(), inp.end()-1, w31.begin(), [](int x) {return ((float)x)/1000.0;});
        
        oS->_inputTensors.clear();
        
        oS->_inputTensors.push_back(Ort::Value::CreateTensor<float>(oS->_memoryInfo, w31.data(), inputTensorSize, inputDims.data(), inputDims.size()));
 
        if(verbose) {cout << "Input tensors number: " << oS->_inputTensors.size() << endl;}

        outputTensors.clear();
        outputTensors.push_back(Ort::Value::CreateTensor<float>(oS->_memoryInfo, outputTensorValues.data(),  outputTensorSize,   outputDims.data(),  outputDims.size()));

        
        // core inference
        oS->_session->Run(Ort::RunOptions{nullptr},
            inputNames.data(),  (oS->_inputTensors).data(),    1,            
            outputNames.data(), outputTensors.data(),   1);
        
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<microseconds>(stop - start);
        cout << "Microseconds: " << duration.count() << endl;
        
        std::cout << outputTensorValues << std::endl;
    }
    exit(0);
}

   //Declare container for input data: std::vector<float> inputTensorValues(inputTensorSize);
        // if(verbose) {for(int bin=0; bin<32; bin++) {cout<< waveform[27][bin] << " ";} cout << endl; }
        // cout << sizeof(waveform[27]) <<endl;
        // vector<float> a[31];
        // for(int bin=0; bin<31; bin++) {inputTensorValues[bin] = (float) waveform[27][bin];} // low-tech way of conversion
//  memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
//            memoryInfo, w31.data(), inputTensorSize, inputDims.data(),         
