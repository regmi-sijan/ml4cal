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

#include "lyra.hpp"
#include "onnxutil.h"



int main(int argc, char* argv[]) {

    bool help               =   false;
    bool verbose            =   false;
    std::string modelfile   =   "tfmodel.onnx";


    auto cli = lyra::cli()
        | lyra::opt(verbose)
            ["-v"]["--verbose"]
            ("verbose" )
        | lyra::opt(modelfile, "model" )
            ["-m"]["--model"]
            ("model")
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

    std::string modelFilepath{modelfile};
    std::string instanceName{"fit"};

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());

    Ort::SessionOptions sessionOptions;

    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);



    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    const char* inputName = session.GetInputName(0, allocator);


    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

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
        std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
        std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
        std::cout << "Input Name: " << inputName << std::endl;
        std::cout << "Input Type: " << inputType << std::endl;
        std::cout << "Input Dimensions: " << inputDims << std::endl;
        std::cout << "Output Name: " << outputName << std::endl;
        std::cout << "Output Type: " << outputType << std::endl;
        std::cout << "Output Dimensions: " << outputDims << std::endl;
    }


    size_t inputTensorSize = vectorProduct(inputDims);
    size_t outputTensorSize = vectorProduct(outputDims);

    std::vector<float> inputTensorValues(inputTensorSize);

    inputTensorValues.assign({1554.0, 1558.0, 1555.0, 1564.0, 1558.0, 1555.0, 1556.0, 1554.0, 1750.0, 2284.0, 2424.0, 2116.0, 1838.0, 1713.0, 1649.0, 1613.0, 1601.0, 1589.0, 1583.0, 1578.0, 1572.0, 1574.0, 1573.0, 1569.0, 1567.0, 1562.0, 1563.0, 1560.0, 1561.0, 1557.0, 1557.0});
    std::cout <<inputTensorValues<< std::endl;

    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));
        
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);            

    std::cout << outputTensorValues << std::endl;

}
