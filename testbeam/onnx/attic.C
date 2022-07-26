Ort::Env create_env(const char* name) {
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, name);
    return env;
}


Ort::Session onnx_session(const char* modelFilepath, const char* envName) {
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, envName);

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, modelFilepath, sessionOptions);
    return session;
}


std::vector<int64_t> onnx_inputDimensions(Ort::Session const* s) {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes        = s->GetInputCount();
    size_t numOutputNodes       = s->GetOutputCount();

    const char* inputName       = s->GetInputName(0, allocator);


    Ort::TypeInfo inputTypeInfo = s->GetInputTypeInfo(0);
    auto inputTensorInfo        = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    inputDims[0] = 1; // fixing feature/bug in ONNX

    return inputDims;
}

// const char* c_str() const;
// std::vector<int64_t> foo = {1,2,3};