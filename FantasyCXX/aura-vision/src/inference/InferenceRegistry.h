#ifndef VISION_INFERENCE_REGISTRY_H
#define VISION_INFERENCE_REGISTRY_H

#include <unordered_map>

#include "AbsPredictor.h"
#include "vision/core/common/VConstants.h"
#include "model_ir/model_info.h"
#include <array>

namespace aura::vision {

using PredictorMap = std::unordered_map<ModelId, std::shared_ptr<AbsPredictor>, EnumClassHash>;

class InferRegistry {
public:
    /**
     * 进行Predictor推理器的注入
     * @param id         ModelId
     * @param predictor Predictor
     */
    static void insert(ModelId id, std::shared_ptr<AbsPredictor> &&predictor);

    /**
     * 进行Predictor推理器的注入
     * @param id         ModelId
     * @param predictor Predictor
     */
    static void insert(int source, ModelId id, std::shared_ptr<AbsPredictor> &&predictor);

    /**
     * 根据Source和ModelId获取对应的sPredictor
     * @param id
     * @return
     */
    static std::shared_ptr<AbsPredictor> get(ModelId id);

    /**
     * 根据Source和ModelId获取对应的sPredictor
     * @param source
     * @param id
     * @return
     */
    static std::shared_ptr<AbsPredictor> get(int source, ModelId id);

    static void clear();
    /**
     * 设置给推理器的指令，具体指令参见：
     * @see VConstant::InferenceCmd
     * @return
     */
    static bool setInferenceCmd(int source, int cmd);

private:
    /**
     * 定义推理器Map的数组结果。数组大小是SOURCE_COUNT + 1。
     * Enum的定义Source的是从SOURCE_UNKNOWN，SOURCE_DMS，SOURCE_OMS，SOURCE_3。
     */
    static std::array<PredictorMap, SOURCE_COUNT + 1> predictorMapArray;
};

} // namespace vision

#endif //VISION_INFERENCE_REGISTRY_H
