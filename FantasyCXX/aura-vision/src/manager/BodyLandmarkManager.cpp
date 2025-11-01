//
// Created by Wang,Bin on 2023/2/9.
//

#include "BodyLandmarkManager.h"
#include "detector/BodyLandmarkDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

BodyLandmarkManager::BodyLandmarkManager() :
        noBodyWindow(SOURCE_UNKNOWN, DEFAULT_WINDOW_LENGTH, DEFAULT_TRIGGER_DUTY_FACTOR, DEFAULT_END_DUTY_FACTOR,
                   F_BODY_NONE) {
    bodyDetector = std::make_shared<BodyLandmarkDetector>();
    setupSlidingWindow();
}

void BodyLandmarkManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    bodyDetector->init(cfg);
    noBodyWindow.setSourceId(cfg->sourceId);
}

void BodyLandmarkManager::deinit() {
    AbsVisionManager::deinit();
    if (bodyDetector != nullptr) {
        bodyDetector->deinit();
        bodyDetector = nullptr;
    }
}

bool BodyLandmarkManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_BODY_LANDMARK);
}

void BodyLandmarkManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_BODY_LANDMARK);

    bodyDetector->detect(request, result);
    // ignore body landmark sliding window
    /*if (result->hasBody()) {
        noBodyWindow.clear();
    } else {
        BodyInfo *body = result->getBodyResult()->pBodyInfos[0];
        if (body) {
            noBodyWindow.update(F_BODY_NONE, &body->noBodyState);
        }
    }*/
}

void BodyLandmarkManager::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = WINDOW_LOWER_FPS; i <= WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * DEFAULT_W_LENGTH_RATIO_1_0)), DEFAULT_W_DUTY_FACTOR};
    }
    noBodyWindow.set_fps_stage_parameters(stageParas);
}

void BodyLandmarkManager::clear() {
    noBodyWindow.clear();
}

void BodyLandmarkManager::onNoBody(VisionRequest *request, VisionResult *result, BodyInfo *body) {
    /*BodyInfo *body = result->getBodyResult()->pBodyInfos[0];
    noBodyWindow.update(F_BODY_NONE, &body->noBodyState);*/
}

REGISTER_VISION_MANAGER("BodyLandmarkManager", ABILITY_BODY_LANDMARK, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<BodyLandmarkManager>());
});

} // namespace aura::vision