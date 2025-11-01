#ifndef VISION_TF_LITE_PREDICTOR_H
#define VISION_TF_LITE_PREDICTOR_H

#include <iostream>
#include <memory>

#include "tensorflow/lite/interpreter.h"

#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"

namespace aura::vision {

class TfLitePredictor : public AbsPredictor {
public:
    ~TfLitePredictor() override = default;
    int init(const ModelInfo& model) override;
    int doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) override;
    int deinit() override;
    bool valid() override;
    std::vector<ModelInput> get_input_desc() const override;

private:
    std::unique_ptr<tflite::Interpreter> _interpreter;
    ModelInfo _model_info;
    bool _inited;
};

template <>
inline std::shared_ptr<AbsPredictor> make_predictor<TF_LITE>(const ModelInfo& model) {
    auto predictor = std::make_shared<TfLitePredictor>();
    if(predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_TF_LITE_PREDICTOR_H