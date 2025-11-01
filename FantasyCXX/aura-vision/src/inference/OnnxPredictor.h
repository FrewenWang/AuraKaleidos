#ifndef VISION_TF_LITE_PREDICTOR_H
#define VISION_TF_LITE_PREDICTOR_H

#include <iostream>
#include <memory>

#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

namespace aura::vision {

class OnnxPredictor : public AbsPredictor {
public:
    OnnxPredictor();

    ~OnnxPredictor() override = default;

    int init(ModelInfo &model) override;

    int doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) override;

    int deinit() override;

    bool valid() override;
    /**
     * ONNX模型推理器的支持的Layout：NCHW
     * @return
     */
    DLayout getSupportedLayout() override;

    std::vector<ModelInput> get_input_desc() const override;

private:
    /** 模型反序列化生成的数据结构 */
    ModelInfo modelInfo;
    bool inited{};
    /**  定义OnnxRuntime Env */
    Ort::Env env;
    /**  定义OnnxRuntime Session */
    Ort::Session session;
    /**  定义OnnxRuntime session_options */
    Ort::SessionOptions session_options;
    /** 定义OnnxRuntime MemoryInfo */
    Ort::MemoryInfo memoryInfo;

    Ort::AllocatorWithDefaultOptions inputAllocator;
    Ort::AllocatorWithDefaultOptions outputAllocator;
    /** 模型推理输入输出的大小 */
    size_t inputCount = 0;
    size_t outputCount = 0;
    /** 模型输入和输出网络层的名称 */
    std::vector<const char *> inputNodeNames = {};
    std::vector<const char *> outputNodeNames = {};
    /** 模型输入和输出网络层的名称 */
    std::vector<std::vector<int64_t>> inputDimsList{};
    std::vector<std::vector<int64_t>> outputDimsList{};
    /** 模型推理的输入数据 */
    std::vector<Ort::Value> ortInputs;

};

template<>
inline std::shared_ptr<AbsPredictor> make_predictor<ONNX>(ModelInfo &model) {
    auto predictor = std::make_shared<OnnxPredictor>();
    if (predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision

#endif //VISION_TF_LITE_PREDICTOR_H