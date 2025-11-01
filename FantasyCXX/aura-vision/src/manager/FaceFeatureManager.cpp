

#include "FaceFeatureManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceFeatureManager";

FaceFeatureManager::FaceFeatureManager() {
    _detector = std::make_shared<FaceFeatureDetector>();
}

void FaceFeatureManager::init(RtConfig* cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
}

void FaceFeatureManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool FaceFeatureManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_FEATURE);
}

void FaceFeatureManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_FEATURE);
    auto ret = _detector->detect(request, result);
    if (ret != V_TO_INT(Error::OK)) {
        VLOGE(TAG, "face_feature source[%d] detect result:%d", mRtConfig->sourceId, ret);
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceFeatureManager", ABILITY_FACE_FEATURE,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceFeatureManager>());
});

} // namespace vision
