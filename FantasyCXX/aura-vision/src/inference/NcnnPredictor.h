#ifndef VISION_NCNN_PREDICTOR_H
#define VISION_NCNN_PREDICTOR_H

#include <iostream>
#include <memory>

#include "net.h" // todo: should refactor to "ncnn/net.h"

#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"

namespace aura::vision {

class NcnnPredictor : public AbsPredictor {
public:
    NcnnPredictor();
    ~NcnnPredictor() override = default;

    int init(const ModelInfo& model) override;
    int doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) override;
    int deinit() override;
    bool valid() override;
    std::vector<ModelInput> get_input_desc() const override;

private:
    ncnn::Net _net;
    ModelInfo _model_info;
    bool _inited;
};

template <>
inline std::shared_ptr<AbsPredictor> make_predictor<NCNN>(const ModelInfo& model) {
    auto predictor = std::make_shared<NcnnPredictor>();
    if (predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_NCNN_PREDICTOR_H
