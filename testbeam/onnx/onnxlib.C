#include <onnxruntime_cxx_api.h>

Ort::Env create_env(const char* name) {
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, name);
    return env;
}

// const char* c_str() const;