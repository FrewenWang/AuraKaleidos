
#include "LivingManager.h"
#include "detector/LivingDetector.h"
#include "util/sliding_window.h"
#include "vision/manager/VisionManagerRegistry.h"
#include <set>

namespace aura::vision {

static const char *TAG = "LivingManager";

LivingStrategy::LivingStrategy(RtConfig *cfg)
    : livingCategoryWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                           AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

LivingStrategy::~LivingStrategy() {}

void LivingStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }
    livingCategoryWindow.set_fps_stage_parameters(stageParas);
}

void LivingStrategy::execute(LivingInfo *category) {
    category->livingType = livingCategoryWindow.update(category->livingTypeSingle);
    if (category->livingType != LivingCategory::F_CATEGORY_CAT && category->livingType != LivingCategory::F_CATEGORY_DOG
        && category->livingType != LivingCategory::F_CATEGORY_BABY) {
        category->livingType = LivingCategory::F_CATEGORY_NONE;
    }
    VLOGI(TAG, "living typeSingle=%d,type=%d", category->livingTypeSingle, category->livingType);
}

void LivingStrategy::clear() {
    livingCategoryWindow.clear();
}

LivingManager::LivingManager() {
    _detector = std::make_shared<LivingRectDetector>();
}

LivingManager::~LivingManager() {
    clear();
    //delete livingStrategy;
}

void LivingManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
    //livingStrategy = new LivingStrategy(cfg);
}

void LivingManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool LivingManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_LIVING_DETECTION);
}

void LivingManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_LIVING_DETECTION);
    _detector->detect(request, result);

    execute_living_strategy(result, livingStrategyMap, mRtConfig);
}

void LivingManager::clear() {
    for (auto &info : livingStrategyMap) {
        if (info.second) {
            info.second->clear();
            LivingStrategy::recycle(info.second);
        }
    }
    livingStrategyMap.clear();
}

REGISTER_VISION_MANAGER("LivingManager", ABILITY_LIVING_DETECTION, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<LivingManager>());
});

} // namespace aura::vision