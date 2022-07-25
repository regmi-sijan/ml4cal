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
        std::cout << "*** Verbose mode selected" << std::endl;
        std::cout << "*** Model file: " << modelfile << std::endl;
        std::cout << "*** ROOT file: " << rootfile << std::endl;
        std::cout << "*** Number of entries to be processed: " << N << std::endl;
    }

    std::string modelFilepath{modelfile};
    std::string instanceName{"fit"};

    // Ort::Env env = create_env(instanceName.c_str());
    // Ort::SessionOptions sessionOptions;
    // sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::Session session = onnx_session(modelFilepath.c_str(), instanceName.c_str());

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes        = session.GetInputCount();
    size_t numOutputNodes       = session.GetOutputCount();

    const char* inputName       = session.GetInputName(0, allocator);


    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo        = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    inputDims[0] = 1;


    const char* outputName = session.GetOutputName(0, allocator);


    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    outputDims[0] = 1;

    if(verbose) {
        std::cout << "*** Number of Input Nodes: "      << numInputNodes    << std::endl;
        std::cout << "*** Number of Output Nodes: "     << numOutputNodes   << std::endl;
        std::cout << "*** Input Name: "                 << inputName        << std::endl;
        std::cout << "*** Input Type: "                 << inputType        << std::endl;
        std::cout << "*** Input Dimensions: "           << inputDims        << std::endl;
        std::cout << "*** Output Name: "                << outputName       << std::endl;
        std::cout << "*** Output Type: "                << outputType       << std::endl;
        std::cout << "*** Output Dimensions: "          << outputDims       << std::endl;
    }


    size_t inputTensorSize  = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);

    //Declare container for input data:
    std::vector<float> inputTensorValues(inputTensorSize);

    // Proceed to read the data from a ROOT tree:
    TFile f(TString(rootfile.c_str()));
    if (f.IsZombie()) { cout << "Error opening file" << endl; exit(-1);}
    if (verbose) { cout << "*** Input ROOT file " << rootfile << " has been opened." << endl;}

    TTree   *tree       = (TTree*)f.Get("trainingtree;1");
    TBranch *branch     = tree->GetBranch("waveform");

    Int_t waveform[64][32];
    branch->SetAddress(&waveform);      // will read into this array
    Long64_t n = branch->GetEntries();  // number of entries in the branch
    if( N==0 || N>n ) { N=n; }          // decide how many to process, 0=all

    if(verbose) {
        std::cout << "*** Number of entries in the file: " << n << std::endl;
        std::cout << "*** Number of entries to be processed: " << N << std::endl;
    }



    // start Ort boilerplate
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // end Ort boilerplace

    for (int i=0; i<N; i++) {
        Int_t m = branch->GetEntry(i);
        // if(verbose) {for(int bin=0; bin<32; bin++) {cout<< waveform[27][bin] << " ";} cout << endl; }

        // cout << sizeof(waveform[27]) <<endl;
        vector<float> a[31];

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
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
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
