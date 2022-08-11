#include <iostream>
#include "onnxlib.h"


void onnx(std::string &modelfile, std::vector<float> &input, int N) {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);


    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "fit");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session* session = new Ort::Session(env, modelfile.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<Ort::Value>     inputTensors, outputTensors;

    // int N = input.size();

    std::vector<int64_t> inputDimsN     = {N,31};
    std::vector<int64_t> outputDimsN    = {N,3};

    std::vector<float>   outputTensorValuesN(N*3);

    inputTensors.push_back (Ort::Value::CreateTensor<float>(memoryInfo, input.data(),               N*31,   inputDimsN.data(),  inputDimsN.size()));
    outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValuesN.data(), N*3,    outputDimsN.data(), outputDimsN.size()));

    std::vector<const char*> inputNames{session->GetInputName(0, allocator)};
    std::vector<const char*> outputNames{session->GetOutputName(0, allocator)};

    session->Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), 1, outputNames.data(), outputTensors.data(), 1);

    for (int i=0; i<N*3; i=i+3) {
        for (int k=0; k<3; k++) {std::cout << outputTensorValuesN[i+k] << " ";}
        std::cout << std::endl;
    }
    

}