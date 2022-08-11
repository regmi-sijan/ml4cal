#include <iostream>
#include "onnxsession.h"

OnnxSession::OnnxSession(const char* modelFilepath, const char* envName, int N) {

    _N = N;

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, envName);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    _session = new Ort::Session(env, modelFilepath, sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // _memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    _numInputNodes              = _session->GetInputCount();
    _numOutputNodes             = _session->GetOutputCount();

    _inputName                  = _session->GetInputName(0, allocator);
    _outputName                 = _session->GetOutputName(0, allocator);

    Ort::TypeInfo inputTypeInfo = _session->GetInputTypeInfo(0);
    auto inputTensorInfo        = inputTypeInfo.GetTensorTypeAndShapeInfo();

    _inputType = inputTensorInfo.GetElementType();

    _inputDimensions = inputTensorInfo.GetShape();
    _inputDimensions[0] = 1; // fixing feature/bug in ONNX

    Ort::TypeInfo outputTypeInfo = _session->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    _outputType = outputTensorInfo.GetElementType();

    _outputDimensions = outputTensorInfo.GetShape();
    _outputDimensions[0] = 1; // fixing feature/bug in ONNX
}

const char* OnnxSession::inputName(void)                    {return _inputName;}
const char* OnnxSession::outputName(void)                   {return _outputName;}

size_t OnnxSession::numInputNodes(void)                     {return _numInputNodes;}
size_t OnnxSession::numOutputNodes(void)                    {return _numOutputNodes;}

std::vector<int64_t>  OnnxSession::inputDimensions(void)    {return _inputDimensions;}
std::vector<int64_t>  OnnxSession::outputDimensions(void)   {return _outputDimensions;}

ONNXTensorElementDataType OnnxSession::inputType(void)      {return _inputType;}
ONNXTensorElementDataType OnnxSession::outputType(void)     {return _outputType;}

void OnnxSession::loadData(std::vector<float> inputData, size_t inputTensorSize) {
    std::cout<<"hello"<<":"<< _memoryInfo << std::endl;
    // _inputTensors.clear();
    for(int i=0; i<31; i++) {std::cout << inputData.data()[i] <<" ";} std::cout << std::endl;
    //std::cout <<  _inputDimensions.data()[1] << ":" << _inputDimensions.size() << ":"<< inputTensorSize<< std::endl;
    _inputTensors.push_back(Ort::Value::CreateTensor<float>(_memoryInfo, inputData.data(), inputTensorSize, _inputDimensions.data(), _inputDimensions.size()));    
}

void prep(Ort::MemoryInfo &memoryInfo, std::vector<float> inputData, std::vector<Ort::Value> &inputTensors, std::vector<int64_t> inputDims) {
    inputTensors.clear();
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputData.data(),  31,   inputDims.data(), inputDims.size()));
}

