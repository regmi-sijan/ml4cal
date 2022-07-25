#include <onnxruntime_cxx_api.h>

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



// const char* c_str() const;