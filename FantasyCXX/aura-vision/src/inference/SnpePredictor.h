#ifndef VISION_SNPE_PREDICTOR_H
#define VISION_SNPE_PREDICTOR_H

#include <iostream>
#include <memory>

#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPE.hpp"

#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"

namespace aura::vision {

class SnpePredictor : public AbsPredictor {
public:
    SnpePredictor();
    ~SnpePredictor() override = default;

    int init(const ModelInfo& model) override;
    int doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) override;
    int deinit() override;
    bool valid() override;
    std::vector<ModelInput> get_input_desc() const override;

private:
    int setRuntime();

    ModelInfo _model_info;
    bool _inited;

    // snpe model
    std::unique_ptr<zdl::SNPE::SNPE> _engine;
    std::unique_ptr<zdl::DlContainer::IDlContainer> _container;

    // snpe input & output
    zdl::DlSystem::StringList _output_tensor_names;
    // if use ITensor
    std::unique_ptr<zdl::DlSystem::ITensor> _input_tensor;
    zdl::DlSystem::TensorMap _output_tensor_map;

    // snpe builder config
    zdl::DlSystem::RuntimeList _runtime_list;
};

template <>
inline std::shared_ptr<AbsPredictor> make_predictor<SNPE>(const ModelInfo& model) {
    auto predictor = std::make_shared<SnpePredictor>();
    if(predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_SNPE_PREDICTOR_H
