#ifndef VISION_PADDLE_LITE_PREDICTOR_H
#define VISION_PADDLE_LITE_PREDICTOR_H

#include <iostream>
#include <memory>

#include "AbsPredictor.h"
#include "paddle_api.h"
#include "vision/core/common/VMacro.h"

namespace aura::vision {

class PaddleLitePredictor : public AbsPredictor {
public:
    ~PaddleLitePredictor() override = default;

    int init(ModelInfo &model) override;

    int doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) override;

    int deinit() override;

    bool valid() override;

    std::vector<ModelInput> get_input_desc() const override;

    DLayout getSupportedLayout() override;

private:
    /**
     * 初始化paddle-lite的config
     */
    paddle::lite_api::MobileConfig _config;
    /**
     * paddle-lite的 predictor
     */
    std::shared_ptr<paddle::lite_api::PaddlePredictor> _predictor;
    ModelInfo _model_info;
    bool _inited;
};

template<>
inline std::shared_ptr<AbsPredictor> make_predictor<PADDLE_LITE>(ModelInfo &model) {
    auto predictor = std::make_shared<PaddleLitePredictor>();
    if (predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_PADDLE_LITE_PREDICTOR_H
