#include "SnpePredictor.h"

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "util/InferenceConverter.hpp"
#include "util/TensorConverter.h"

namespace aura::vision {

SnpePredictor::SnpePredictor() : _inited(false) {}

static const char* TAG = "SnpePredictor(2000)";

int SnpePredictor::init(const ModelInfo& model) {
    mPerfTag = "[Predictor]: [Snpe] " + std::to_string(model.id);
    _model_info = model;
    if (_model_info.blobs.size() != 1) {
        VLOGE(TAG, "Predictor model mem error, one blob needed! error model info:%s", _model_info.to_string().c_str());
        VA_RET(Error::MODEL_INIT_ERR);
    }
    VLOGI(TAG, "Predictor init model: %s", _model_info.port_desc.ability().c_str());

    // 1.set runtime
    int ret = setRuntime();
    VA_CHECK_RET_MSG(ret, Error::MODEL_INIT_ERR, zdl::DlSystem::getLastErrorString());

    // 2. load model
    auto* dlc_mem = static_cast<const uint8_t*>(model.blobs[0].data);
    auto size = static_cast<const size_t>(model.blobs[0].len);
    _container = zdl::DlContainer::IDlContainer::open(dlc_mem, size);
    VA_CHECK_NULL_RET_MSG(_container.get(), Error::MODEL_INIT_ERR, zdl::DlSystem::getLastErrorString());

    // 3. build engine
    auto output = _model_info.port_desc.snpe_output_size() > 0 ? _model_info.port_desc.snpe_output() : _model_info.port_desc.output();
    for (const auto& out : output) {
        _output_tensor_names.append(out.c_str());
        VLOGI(TAG, "add output tensor: %s", out.c_str());
    }
    zdl::SNPE::SNPEBuilder snpe_builder(_container.get());
    _engine = snpe_builder.setOutputTensors(_output_tensor_names)
                            .setRuntimeProcessorOrder(_runtime_list)
                            .build();
    VA_CHECK_NULL_RET_MSG(_engine, Error::MODEL_INIT_ERR, zdl::DlSystem::getLastErrorString());

    _inited = true;
    VA_RET(Error::OK);
}

int SnpePredictor::deinit() {
    _inited = false;
}

int SnpePredictor::doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) {
    outputs.clear();

    V_CHECK_COND(!_inited, Error::MODEL_UNINITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UNINITED, "Error: predictor inputs are empty!");

    // handle inputs TODO error when use snpe TensorMap do inference, consider num of input tensor is 1
    Tensor src_tensor = inputs[0];
    if (src_tensor.layout == NCHW) {
        src_tensor = src_tensor.change_layout(NHWC);
    }
    _input_tensor = TensorConverter::convert_to<std::unique_ptr<zdl::DlSystem::ITensor>>(src_tensor, true);

    // handle outputs
    bool ret = _engine->execute(_input_tensor.get(), _output_tensor_map);
    // early stop when error
    V_CHECK_COND(!ret, Error::INFER_ERR, zdl::DlSystem::getLastErrorString());


    auto output = _model_info.port_desc.snpe_output_size() > 0 ? _model_info.port_desc.snpe_output() : _model_info.port_desc.output();
    for (const auto& out : output) {
        auto* output_tensor_ptr = _output_tensor_map.getTensor(out.c_str());
        VA_CHECK_NULL_RET_MSG(output_tensor_ptr, Error::INFER_ERR, "predictor collect output tensor ERROR");
        Tensor out_tensor = TensorConverter::convert_from<zdl::DlSystem::ITensor>(*output_tensor_ptr, true);
        outputs.emplace_back(out_tensor);
    }

    V_RET(Error::OK);
}

bool SnpePredictor::valid() {
    return _inited;
}

std::vector<ModelInput> SnpePredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto& input : _model_info.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

int SnpePredictor::setRuntime(){
    Device device = _model_info.device;
    DType dtype = _model_info.dtype;
    zdl::DlSystem::Runtime_t runtime_t;

    if (device == GPU) {
        if (dtype == FP16) {
            runtime_t = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
        } else {
            runtime_t = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
        }
    } else if (device == DSP) {
        runtime_t = zdl::DlSystem::Runtime_t::DSP;
    } else {
        runtime_t = zdl::DlSystem::Runtime_t::CPU;
    }

    const char* runtime_string = zdl::DlSystem::RuntimeList::runtimeToString(runtime_t);

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_t) ||
        (device == GPU && !zdl::SNPE::SNPEFactory::isGLCLInteropSupported())) {
        VLOGE(TAG, "SNPE runtime %s not support", runtime_string);
        VA_RET(Error::MODEL_INIT_ERR);
    }

    VLOGD(TAG, "SNPE model init, using runtime %s", runtime_string);

    // _runtime_list : runtime order list
    _runtime_list.add(runtime_t);
    VA_RET(Error::OK);
}

} // namespace vision