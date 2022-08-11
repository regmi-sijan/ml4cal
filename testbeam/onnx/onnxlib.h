#include <onnxruntime_cxx_api.h>

#ifndef ONNX_SESSION_LIB_H
#define ONNX_SESSION_LIB_H

class OnnxSession {
    public:
        OnnxSession(const char* modelFilepath, const char* envName, int N);
        Ort::Session* _session;

        const char* inputName(void);
        const char* outputName(void);

        size_t numInputNodes(void);
        size_t numOutputNodes(void);

        std::vector<int64_t>  inputDimensions(void);
        std::vector<int64_t>  outputDimensions(void);

        ONNXTensorElementDataType inputType(void);
        ONNXTensorElementDataType outputType(void);

        Ort::MemoryInfo _memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        // ----------------------------------
        std::vector<Ort::Value>  _inputTensors;
        
        void loadData(std::vector<float> input, size_t inputTensorSize);
    
    private:
        int                         _N;

        const char*                 _inputName;
        const char*                 _outputName;

        size_t                      _numInputNodes;
        size_t                      _numOutputNodes;

        std::vector<int64_t>  _inputDimensions;
        std::vector<int64_t>  _outputDimensions;

        ONNXTensorElementDataType   _inputType;
        ONNXTensorElementDataType   _outputType;
};

void prep(Ort::MemoryInfo &memoryInfo, std::vector<float> inputData, std::vector<Ort::Value>  &inputTensors, std::vector<int64_t> inputDims);

#endif
