#ifndef ONNX_SESSION_LIB_H
#define ONNX_SESSION_LIB_H

class OnnxSession {
    public:
        OnnxSession(const char* modelFilepath, const char* envName);
        Ort::Session* _session;

        const char* inputName(void);
        const char* outputName(void);

        size_t numInputNodes(void);
        size_t numOutputNodes(void);

        ONNXTensorElementDataType inputType(void);
        ONNXTensorElementDataType outputType(void);

        std::vector<int64_t>  _inputDimensions;
        std::vector<int64_t>  _outputDimensions;

    private:
        const char* _inputName;
        const char* _outputName;

        size_t _numInputNodes;
        size_t _numOutputNodes;

        ONNXTensorElementDataType _inputType;
        ONNXTensorElementDataType _outputType;
};

#endif
