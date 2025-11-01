
#include "GestureRectManager.h"
#include "detector/GestureRectDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {
GestureRectManager::GestureRectManager() = default;

GestureRectManager::~GestureRectManager() { _hand_lists.clear(); }

void GestureRectManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    _detector = std::make_shared<GestureRectDetector>();
    _hand_lists.resize(V_TO_INT(mRtConfig->gestureMaxCount));
    _detector->init(cfg);
}

void GestureRectManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool GestureRectManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_GESTURE_RECT);
}

void GestureRectManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_GESTURE_RECT);
    _detector->detect(request, result);
}

REGISTER_VISION_MANAGER("GestureRectManager", ABILITY_GESTURE_RECT, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<GestureRectManager>());
});

} // namespace vision