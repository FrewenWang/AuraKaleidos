#ifndef VISION_CUSTOMIZE_PREDICTOR_H
#define VISION_CUSTOMIZE_PREDICTOR_H

#include <iostream>
#include <memory>

#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"

namespace aura::vision {

class CustomizePredictor : public AbsPredictor {
public:
    CustomizePredictor();
    ~CustomizePredictor() override = default;

    int init(ModelInfo& model) override;
    int doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) override;
    int deinit() override;
    bool valid() override;
    std::vector<ModelInput> get_input_desc() const override;
    const char* get_customize_param_data();
    int get_customize_param_size();
    DLayout getSupportedLayout();

private:
    ModelInfo _model_info;
    bool _inited;
};

template <>
inline std::shared_ptr<AbsPredictor> make_predictor<CUSTOMIZE>(ModelInfo& model) {
    auto predictor = std::make_shared<CustomizePredictor>();
    if (predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_CUSTOMIZE_PREDICTOR_H
