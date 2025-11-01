#include "AbsPredictor.h"

// 当新增 predictor 时，需要在这包含头文件
#include "CustomizePredictor.h"
#ifdef BUILD_NCNN
#include "NcnnPredictor.h"
#endif
#ifdef USE_PADDLE_LITE
#include "PaddleLitePredictor.h"
#endif
#ifdef USE_TF_LITE
#include "tf_lite_predictor.h"
#endif
#ifdef USE_SNPE
#include "snpe_predictor.h"
#endif
#ifdef USE_QNN
#include "QnnPredictor.h"
#endif
#ifdef USE_ONNX
#include "OnnxPredictor.h"
#endif

namespace aura::vision {

template <InferType ...> struct InfeTypeList {};

// Default behavior when no infer_type is matched with the model_info
std::shared_ptr<AbsPredictor> do_create_predictor(ModelInfo& model, InfeTypeList<>) {
    VLOGE("AbsPredictor", "init predictor error");
    return nullptr;
}

// Loop to find the matched InferType
template <InferType Type1, InferType ...TypeN>
std::shared_ptr<AbsPredictor> do_create_predictor(ModelInfo& model, InfeTypeList<Type1, TypeN...>) {
    if (model.infer_type != Type1) {
        return do_create_predictor(model, InfeTypeList<TypeN...>());
    }
    return make_predictor<Type1>(model);
}

template <InferType ...TypeN>
std::shared_ptr<AbsPredictor> do_create_predictor(ModelInfo& model) {
    return do_create_predictor(model, InfeTypeList<TypeN...>());
}

// 当新增 predictor 时，需要在模板参数列表中增加对应的 predictor 的枚举定义
std::shared_ptr<AbsPredictor> InferenceFactory::create_predictor(ModelInfo &model) {
    return do_create_predictor<NCNN, PADDLE_LITE, TNN, SNPE, TF_LITE, CUSTOMIZE, QNN, ONNX>(model);
}


AbsPredictor::AbsPredictor() {
    mPerfTag = "[Predictor]:";
}

int AbsPredictor::predict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) {
    return doPredict(inputs, outputs, perf);
}

int AbsPredictor::predict(std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs, PerfUtil *perf) {
    return doPredict(inputs, outputs, perf);
}

bool AbsPredictor::onInferenceCmd(int cmd) {
    return false;
}

} // namespace aura::vision