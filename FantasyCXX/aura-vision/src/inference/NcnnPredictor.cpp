#include "NcnnPredictor.h"

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "util/InferenceConverter.hpp"
#include "util/TensorConverter.h"

namespace aura::vision {

static const char* TAG = "NcnnPredictor(1000)";

NcnnPredictor::NcnnPredictor() : _inited(false) {}

int NcnnPredictor::init(const ModelInfo& model) {
    mPerfTag = "[Predictor]: [Ncnn] " + std::to_string(model.id);
    _model_info = model;
    if (model.blobs.size() < 2) {
        VLOGE(TAG, "Predictor model mem error, two blobs are needed! error model info:%s", model.to_string().c_str());
        VA_RET(Error::MODEL_INIT_ERR);
    }

    auto* proto_mem = (const char*)model.blobs[0].data;
    auto* model_mem = (const unsigned char*)model.blobs[1].data;

    if (_net.load_param_mem(proto_mem) != 0) {
        VLOGE(TAG, "Predictor init param FAILED");
        _inited = false;
        VA_RET(Error::MODEL_INIT_ERR);
    }

    if (_net.load_model(model_mem) <= 0) {
        VLOGE(TAG, "Predictor init weights FAILED");
        _inited = false;
        VA_RET(Error::MODEL_INIT_ERR);
    }

    _inited = true;
    VA_RET(Error::OK);
}

int NcnnPredictor::deinit() {
    _net.clear();
    _inited = false;
    VA_RET(Error::OK);
}

int NcnnPredictor::doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) {
    outputs.clear();

    V_CHECK_COND(!_inited, Error::MODEL_UNINITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UNINITED, "Error: predictor inputs are empty!");

    ncnn::Extractor extractor = _net.create_extractor();
    extractor.set_light_mode(true);
    if (_model_info.enable_omp) {
        // NCNN Predictor的 目前无法拿到RuntimeConfig对象，暂时写死
        // auto num_threads = VA_TO_INT(config::_s_num_threads);
        auto num_threads = 4;
        extractor.set_num_threads(num_threads);
        VLOGI(TAG, "Predictor [%s] enables omp, num_threads=%d",
                TagIdConverter::get_tag<ModelId>(_model_info.id).c_str(), num_threads);
    }

    // handle inputs
    int index = 0;
    for (const auto& in : _model_info.port_desc.input()) {
        // convert input tensor to ncnn mat
        auto ncnn_input = TensorConverter::convert_to<ncnn::Mat>(inputs[index]);
        extractor.input(in.name().c_str(), ncnn_input);
        index++;
    }

    // handle outputs
    PERF_TICK(perf, mPerfTag)
    for (const auto& out : _model_info.port_desc.output()) {
        ncnn::Mat ncnn_output;
        auto ret = extractor.extract(out.c_str(), ncnn_output);
        if (ret != 0) {
            VLOGE(TAG, "Predictor[%s] extract %s ERROR, code=%d",
                    TagIdConverter::get_tag<ModelId>(_model_info.id).c_str(), out.c_str(), ret);
            VA_RET(Error::INFER_ERR); // early stop when error
        }
        outputs.emplace_back(TensorConverter::convert_from<ncnn::Mat>(ncnn_output, true));
    }
    PERF_TOCK(perf, mPerfTag)

    VA_RET(Error::OK);
}

bool NcnnPredictor::valid() {
    return _inited;
}

std::vector<ModelInput> NcnnPredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto& input : _model_info.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

} // namespace vision