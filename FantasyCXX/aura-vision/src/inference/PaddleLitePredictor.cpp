#include "PaddleLitePredictor.h"

#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/util/log.h"
#include "util/InferenceConverter.hpp"
#include "util/TensorConverter.h"
#include "util/DebugUtil.h"

#ifdef BUILD_IOS
#include "paddle_use_ops.h"
#include "paddle_use_kernels.h"
#endif

namespace aura::vision {

static const char* TAG = "PaddleLitePredictor(1001)";

int PaddleLitePredictor::init(ModelInfo &model) {
    _model_info = model;
    if (model.blobs.empty()) {
        VLOGE(TAG, "Predictor model mem error, no blob found! error model info:%s", model.to_string().c_str());
        _inited = false;
        V_RET(Error::MODEL_INIT_ERR);
    }

    auto *mem = static_cast<const char *>(model.blobs[0].data);
    std::string model_buffer(mem, static_cast<size_t>(model.blobs[0].len));
    _config.set_model_from_buffer(model_buffer);
    _config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
    if (_model_info.enable_omp) {
        auto num_threads = V_TO_INT(5);
        _config.set_threads(num_threads);
        VLOGI(TAG, "Predictor [%s] enables omp, num_threads=%d",
              InferenceConverter::get_tag<ModelId>(_model_info.id).c_str(), num_threads);
    }
    // Create PaddlePredictor by MobileConfig
    _predictor = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(_config);
    _inited = true;
    V_RET(Error::OK);
}

int PaddleLitePredictor::deinit() {
    _predictor.reset();
    V_RET(Error::OK);
}

int PaddleLitePredictor::doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) {
    outputs.clear();
    V_CHECK_COND(!_inited, Error::MODEL_UN_INITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UN_INITED, "Error: predictor inputs are empty!");

    // Prepare input data
    for (int i = 0; i < _model_info.port_desc.input_size(); ++i) {
        auto input = inputs[i];
        // PaddleLitePredictor模型要求输出格式为NCHW。
        if (input.dLayout != NCHW) {
            input = input.changeLayout(NCHW);
        }
        // inputs的data输入的数据是 int8 的数据
        const auto &mat_src = vision::TensorConverter::convert_to<cv::Mat>(input);
        std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(_predictor->GetInput(i)));
        input_tensor->Resize({1, input.c, input.h, input.w});
        memcpy(input_tensor->mutable_data<float>(), input.data, input.len());

        // cv::Mat matConveted;
        // if (input.dType != FP32) {
        //     if (input.c == 1) {
        //         mat_src.convertTo(matConveted, CV_32FC1);
        //     } else if (input.c == 3) {
        //         mat_src.convertTo(matConveted, CV_32FC3);
        //     } else {
        //         throw std::runtime_error("The tensor channel number  is not supported in resizeNoNormalize()");
        //     }
        // } else {
        //     matConveted = mat_src;
        // }
        // auto dst = vision::TensorConverter::convert_from<cv::Mat>(matConveted, true);
        // // 将转化之后的数据送入 PaddleLitePredictor进行推理
        // std::unique_ptr<paddle::lite_api::Tensor> input_tensor(std::move(_predictor->GetInput(i)));
        // input_tensor->Resize({1, dst.c, dst.h, dst.w});
        // memcpy(input_tensor->mutable_data<float>(), dst.data, dst.len());
        // DBG_PRINT_ARRAY((float *)dst.data,20,"paddle-lite_prepare_data");
    }
    // Run predictor
    _predictor->Run();

    // handle outputs
    for (int i = 0; i < _model_info.port_desc.output_size(); ++i) {
        std::unique_ptr<const paddle::lite_api::Tensor> pd_output(std::move(_predictor->GetOutput(i)));
        outputs.emplace_back(TensorConverter::convert_from<paddle::lite_api::Tensor>(*pd_output, true));
    }

    V_RET(Error::OK);
}

bool PaddleLitePredictor::valid() {
    return _inited;
}

std::vector<ModelInput> PaddleLitePredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto& input : _model_info.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

DLayout PaddleLitePredictor::getSupportedLayout() {
    return NCHW;
}

} // namespace vision
