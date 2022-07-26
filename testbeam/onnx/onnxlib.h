#ifndef ONNX_SESSION_LIB_H
#define ONNX_SESSION_LIB_H

class OnnxSession {
    public:
        OnnxSession(const char* modelFilepath, const char* envName);
        Ort::Session* _session;
        std::vector<int64_t>  _inputDimensions;
        std::vector<int64_t>  _outputDimensions;

        ONNXTensorElementDataType _inputType;
        ONNXTensorElementDataType _outputType;

        const char* _inputName;
        const char* _outputName;

        size_t _numInputNodes;
        size_t _numOutputNodes;        
};

#endif
