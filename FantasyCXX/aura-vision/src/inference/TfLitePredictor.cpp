#include "TfLitePredictor.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "util/InferenceConverter.hpp"
#include "util/TensorConverter.h"
#include "util/math_utils.h"

namespace aura::vision {

static const char* TAG = "Predictor(1003)";

int TfLitePredictor::init(const ModelInfo& model) {
    mPerfTag = "[Predictor]: [TfLite] " + std::to_string(model.id);
    _model_info = model;
    if (model.blobs.empty()) {
        VLOGE(TAG, "Predictor model mem error, no blob found! error model info:%s", model.to_string().c_str());
        _inited = false;
        VA_RET(Error::MODEL_INIT_ERR);
    }

    // load model
    auto* mem = static_cast<const char*>(model.blobs[0].data);
    auto mem_len = model.blobs[0].len;
    VLOGD(TAG, "tflite model_len=%d", mem_len);
    std::unique_ptr<tflite::FlatBufferModel> tf_model = tflite::FlatBufferModel::BuildFromBuffer(mem, mem_len);
    if (tf_model == nullptr) {
        VLOGE(TAG, "Predictor model load error: model is empty");
        _inited = false;
        V_RET(Error::MODEL_INIT_ERR);
    }

    // build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*tf_model, resolver)(&_interpreter);
    if (_interpreter == nullptr) {
         VLOGE(TAG, "Predictor model load error: build interpreter failed");
        _inited = false;
        V_RET(Error::MODEL_INIT_ERR);
    }

    // num of threads
    if (_model_info.enable_omp) {
        auto num_threads = VA_TO_INT(RtConfig::get_config(MAX_NUM_THREADS));
        _interpreter->SetNumThreads(num_threads);
        VLOGI(TAG, "Predictor [%s] enables omp, num_threads=%d",
            TagIdConverter::get_tag<ModelId>(_model_info.id).c_str(), num_threads);
    }

    // allocate tensor
    _interpreter->AllocateTensors();
    _inited = true;
    V_RET(Error::OK);
}

int TfLitePredictor::deinit() {
    _interpreter.reset();
    V_RET(Error::OK);
}

int TfLitePredictor::doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) {
    outputs.clear();

    V_CHECK_COND(!_inited, Error::MODEL_UNINITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UNINITED, "Error: predictor inputs are empty!");

    // handle inputs
    for (int i = 0; i < _model_info.port_desc.input_size(); ++i) {
        auto input = inputs[i];
        float* in_ptr = _interpreter->typed_input_tensor<float>(i);
        memcpy(in_ptr, input.data, input.len());
    }

    _interpreter->Invoke();

    // handle outputs
    auto tf_out_indices = _interpreter->outputs();
    for (int i = 0; i < _model_info.port_desc.output_size(); ++i) {
        const auto& tf_out_tensor = _interpreter->tensor(tf_out_indices[i]);
        outputs.emplace_back(TensorConverter::convert_from<TfLiteTensor>(*tf_out_tensor, true));
    }

    VA_RET(Error::OK);
}

bool TfLitePredictor::valid() {
    return _inited;
}

std::vector<ModelInput> TfLitePredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto& input : _model_info.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

} // namespace vision