
#include "GestureLandmarkManager.h"
#include "detector/GestureLandmarkDetector.h"
#include "vision/util/log.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {
GestureLandmarkManager::GestureLandmarkManager() {
    _detector = std::make_shared<GestureLandmarkDetector>();
}

void GestureLandmarkManager::init(RtConfig* cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
}

void GestureLandmarkManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool GestureLandmarkManager::preDetect(VisionRequest *request, VisionResult *result) {
    bool checkGestureRect = VA_GET_DETECTED(ABILITY_GESTURE_RECT);
    V_CHECK_COND(!checkGestureRect, Error::PREPARE_ERR, "Gesture Landmark would be scheduled after GestureRect");
    VA_CHECK_DETECTED(ABILITY_GESTURE_LANDMARK);
}

void GestureLandmarkManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_GESTURE_LANDMARK);
    _detector->detect(request, result);
}

REGISTER_VISION_MANAGER("GestureLandmarkManager", ABILITY_GESTURE_LANDMARK,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<GestureLandmarkManager>());
});

} // namespace vision