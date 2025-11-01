
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "model_ir/model_info.h"
#include "vision/core/common/VTensor.h"
#include "vision/util/log.h"
#include "vision/util/PerfUtil.h"
#include "vision/core/common/VTensor.h"
#include "opencv2/core/core.hpp"

namespace aura::vision {

/**
 * @brief Base class of predictor
 */
class AbsPredictor {
public:

    AbsPredictor();

    /// Init predictor
    virtual int init(ModelInfo& model) = 0;
    /**
     * 模型推理器的推理逻辑
     * @param inputs  输入TensorArray
     * @param outputs 输出TensorArray
     * @param perf   性能统计工具
     * @return   返回对应错误码  具体参见： VConstant::Error
     */
    int predict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf);
    /**
     * 模型推理器的推理逻辑
     * @param inputs  输入TensorArray
     * @param outputs 输出TensorArray
     * @param perf   性能统计工具
     * @return   返回对应错误码  具体参见： VConstant::Error
     */
    virtual int doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) = 0;

    int predict(std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs, PerfUtil *perf);

    virtual int doPredict(std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs, PerfUtil* perf) { return 0; };

    /// Deinit predictor
    virtual int deinit() = 0;

    /// Whether the predictor is valid, usually it is invalid after deinited
    virtual bool valid() = 0;

    /// get model input description
    virtual std::vector<ModelInput> get_input_desc() const = 0;

    virtual ~AbsPredictor() = default;

    virtual DLayout getSupportedLayout() = 0;
    /**
     * 接收到推理器设置的的指令
     * @param cmd
     * @see VConstant::InferenceCmd
     * @return
     */
    virtual bool onInferenceCmd(int cmd);

    std::string mPerfTag;
};

/**
 * @brief Factory of creating predictors
 */
class InferenceFactory {
public:
    static std::shared_ptr<AbsPredictor> create_predictor(ModelInfo& model);
};

template <InferType type>
inline std::shared_ptr<AbsPredictor> make_predictor(ModelInfo& model) {
    VLOGE("InferenceFactory", "Inference type %d unsupported!", type);
    return nullptr;
}

#define MAKE_PREDICTOR(predictor, source, id)               \
do {                                                        \
    if (!predictor || !predictor->valid()) {                \
        predictor = InferRegistry::get(source,id);          \
    }                                                       \
} while(0)

} // namespace vision
