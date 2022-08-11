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
#include "onnxsession.h"
#include "onnxutil.h"


using namespace std::chrono;



int main(int argc, char* argv[]) {

    bool help               = false;
    bool verbose            = false;
    bool output             = false;

    int N                   = 0;
    int channel             = 27;

    std::string modelfile   = "tfmodel.onnx";
    std::string rootfile    = "rootfile.root";

    auto cli = lyra::cli()
        | lyra::opt(verbose)
            ["-v"]["--verbose"]
            ("Verbose mode")
        | lyra::opt(output)
            ["-o"]["--output"]
            ("Print inference output")            
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
            ("Caloriemter channel")
        | lyra::help(help);

    auto result = cli.parse( { argc, argv } );
    if( !result ) {std::cerr << "Error in command line: " << result.errorMessage() << std::endl;exit(1);}

    if(help) {std::cout << cli << std::endl; exit(0);}

    if(verbose) {std::cout << "*** Verbose mode selected" << std::endl << "*** Model file: " << modelfile << std::endl << "*** ROOT file: " << rootfile << std::endl;}
 
    std::string instanceName{"fit"};

    OnnxSession* oS = new OnnxSession(modelfile.c_str(), instanceName.c_str(), 10);

    std::vector<int64_t> inputDims = oS->inputDimensions(), outputDims = oS->outputDimensions();


    if(verbose) {
        std::cout << "*** Input Nodes:\t"   << oS->numInputNodes()  << ",\t\t Output Nodes: "  << oS->numOutputNodes() << std::endl;
        std::cout << "*** Input Name:\t\t"  << oS->inputName()      << ",\t Input Type: "      << oS->inputType()  << ",\t Input Dimensions:\t"  << inputDims    << std::endl;
        std::cout << "*** Output Name:\t"   << oS->outputName()     << ",\t Output Type: "     << oS->outputType() << ",\t Output Dimensions:\t" << outputDims   << std::endl;
    }


    size_t inputTensorSize = vectorProduct(inputDims), outputTensorSize = vectorProduct(outputDims);

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

    std::vector<const char*>    inputNames{oS->inputName()}, outputNames{oS->outputName()};
    std::vector<Ort::Value>     inputTensors, outputTensors;


    std::vector<float> w31(31); // 31 bins in the test beam data; was inputTensorSize
    std::vector<float> input{}; //


    std::vector<int64_t> inputDimsN     = {N,31};
    std::vector<int64_t> outputDimsN    = {N,3};

    std::vector<float>   outputTensorValuesN(N*3);


    for (int i=0; i<N; i++) {
        Int_t m = branch->GetEntry(i);

        std::vector<int> inp;
        inp.insert(inp.begin(), std::begin(waveform[channel]), std::end(waveform[channel]));
        std::transform(inp.begin(), inp.end()-1, w31.begin(), [](int x) {return ((float)x)/1000.0;});
        input.insert(input.end(), w31.begin(), w31.end());
    }

    std::cout << "Input size: " << input.size() << std::endl;

    auto start = chrono::high_resolution_clock::now();
    inputTensors.push_back (Ort::Value::CreateTensor<float>(oS->_memoryInfo, input.data(),               N*31,   inputDimsN.data(),  inputDimsN.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(oS->_memoryInfo, outputTensorValuesN.data(), N*3,    outputDimsN.data(), outputDimsN.size()));

    oS->_session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<microseconds>(stop - start);
    cout << "Microseconds: " << duration.count() << endl;

    if(output) {
        for (int i=0; i<N*3; i=i+3) {for (int k=0; k<3; k++) {cout << outputTensorValuesN[i+k] << " ";} cout << endl;}
    }

    // std::cout << "Output size:" << outputTensorValuesN.size() << std::endl;


    exit(0);
}
    
    // ---------------------------------------------------------------------
    // ATTIC:
    //
    // Example:
    // std::array<std::array<int, 3>, 3> arr = {{{5, 8, 2}, {8, 3, 1}, {5, 3, 9}}};

    // std::array<std::array<float,31>, 2> arr = {{
    //     {1554.0, 1558.0, 1555.0,  1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0},
    //     {1554.0, 1558.0, 1555.0,  1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0}
    // }};
    
    // for (auto &row: arr) {
    //     for (auto &i: row) {
    //         std::cout << i << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::vector<float> arr = {
    //     1554.0, 1558.0, 1555.0, 1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0,
    //     1554.0, 1558.0, 1555.0, 1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0,
    //     1552.0, 1550.0, 1552.0, 1552.0, 1553.0, 1550.0, 1551.0, 1554.0, 1551.0, 1551.0, 1582.0, 2617.0, 4401.0, 4371.0, 3194.0, 2360.0, 2013.0, 1844.0, 1743.0, 1687.0, 1658.0, 1643.0, 1630.0, 1617.0, 1610.0, 1602.0, 1598.0, 1594.0, 1585.0, 1577.0
    // };


    // inputTensors.push_back (Ort::Value::CreateTensor<float>(memoryInfo, arr.data(),                 N*31,   inputDimsN.data(),  inputDims.size()));
    // outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValuesN.data(), N*3,    outputDimsN.data(), outputDimsN.size()));
    // oS->_session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);
    // std::cout << outputTensorValuesN << std::endl;
