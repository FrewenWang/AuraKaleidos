

#include "FaceRectManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

FaceRectManager::FaceRectManager() {
    _detector = std::make_shared<FaceRectDetector>();
}

void FaceRectManager::init(RtConfig* cfg) {
    mRtConfig = cfg;
   _detector->init(cfg);
}

void FaceRectManager::deinit() {
   AbsVisionManager::deinit();
   if (_detector != nullptr) {
       _detector->deinit();
       _detector = nullptr;
   }
}

bool FaceRectManager::preDetect(VisionRequest *request, VisionResult *result) { VA_CHECK_DETECTED(ABILITY_FACE_RECT);
}

void FaceRectManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_RECT);

    _detector->detect(request, result);
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceRectManager", ABILITY_FACE_RECT,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceRectManager>());
});

} // namespace vision