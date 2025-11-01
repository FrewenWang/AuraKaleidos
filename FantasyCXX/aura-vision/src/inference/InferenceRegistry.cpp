#include "InferenceRegistry.h"

namespace aura::vision {

std::array<PredictorMap, SOURCE_COUNT + 1> InferRegistry::predictorMapArray;

void InferRegistry::insert(ModelId id, std::shared_ptr<AbsPredictor> &&predictor) {
    // _predictors[id] = predictor;
}

void InferRegistry::insert(int source, ModelId id, std::shared_ptr<AbsPredictor> &&predictor) {
    PredictorMap &predictorMap = predictorMapArray[source];
    predictorMap[id] = predictor;
}

std::shared_ptr<AbsPredictor> InferRegistry::get(ModelId id) {
    // if (_predictors.find(id) != _predictors.end()) {
    //     return _predictors[id];
    // }
    return nullptr;
}

std::shared_ptr<AbsPredictor> InferRegistry::get(int source, ModelId id) {
    auto predictorMap = predictorMapArray[source];
    if (predictorMap.find(id) != predictorMap.end()) {
        return predictorMap[id];
    }
    return nullptr;
}

void InferRegistry::clear() {
    for (int i = 0; i < SOURCE_COUNT; ++i) {
        PredictorMap &predictorMap = predictorMapArray[i];
        for (auto &p: predictorMap) {
            p.second->deinit();
        }
        predictorMap.clear();
    }
}

bool InferRegistry::setInferenceCmd(int source, int cmd) {
    PredictorMap &predictorMap = predictorMapArray[source];
    if (predictorMap.empty()) {
        VLOGW("", "setInferenceCmd failed cause predictorMap is empty");
        return false;
    }
    for (auto &p : predictorMap) {
        p.second->onInferenceCmd(cmd);
    }
    return true;
}

} // namespace vision