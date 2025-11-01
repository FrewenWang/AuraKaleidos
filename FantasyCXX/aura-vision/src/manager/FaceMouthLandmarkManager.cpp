#include "FaceMouthLandmarkManager.h"
#include "detector/FaceMouthLandmarkDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

FaceMouthLandmarkManager::FaceMouthLandmarkManager() {
    detector = std::make_shared<FaceMouthLandmarkDetector>();
}

void FaceMouthLandmarkManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceMouthLandmarkManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceMouthLandmarkManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_MOUTH_LANDMARK);
}

void FaceMouthLandmarkManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_MOUTH_LANDMARK);
    detector->detect(request, result);
}

void FaceMouthLandmarkManager::clear() {

}

REGISTER_VISION_MANAGER("FaceMouthLandmarkManager", ABILITY_FACE_MOUTH_LANDMARK, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceMouthLandmarkManager>());
});

} // namespace vision