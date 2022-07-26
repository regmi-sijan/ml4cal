#include <onnxruntime_cxx_api.h>
#include "onnxlib.h"

OnnxSession::OnnxSession(const char* modelFilepath, const char* envName) {
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, envName);
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _session = new Ort::Session(env, modelFilepath, sessionOptions);


    Ort::AllocatorWithDefaultOptions allocator;

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
