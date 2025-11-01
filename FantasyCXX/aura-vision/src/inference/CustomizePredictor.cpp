#include "CustomizePredictor.h"

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "util/InferenceConverter.hpp"
#include "util/TensorConverter.h"

namespace aura::vision {

static const char* TAG = "Predictor(3000)";

CustomizePredictor::CustomizePredictor() : _inited(false) {}

int CustomizePredictor::init(ModelInfo& model) {
    _model_info = model;
    if (model.blobs.size() < 1) {
        VLOGE(TAG, "Predictor model mem error, one blobs are needed! error model info:%s", model.to_string().c_str());
        V_RET(Error::MODEL_INIT_ERR);
    }

    _inited = true;
    V_RET(Error::OK);
}

int CustomizePredictor::deinit() {
    _inited = false;
    V_RET(Error::OK);
}

int CustomizePredictor::doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) {
    V_RET(Error::OK);
}

bool CustomizePredictor::valid() {
    return _inited;
}

std::vector<ModelInput> CustomizePredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto& input : _model_info.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

const char* CustomizePredictor::get_customize_param_data() {
    return (const char*)_model_info.blobs[0].data;
}

int CustomizePredictor::get_customize_param_size() {
    return _model_info.blobs[0].len;
}

DLayout CustomizePredictor::getSupportedLayout() {
    return NHWC;
}

} // namespace vision
