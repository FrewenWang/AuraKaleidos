

#include "ImageBrightnessManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "ImageBrightnessManager";

ImageBrightnessManager::ImageBrightnessManager() {
    mDetector = std::make_shared<ImageBrightnessDetector>();
}

ImageBrightnessManager::~ImageBrightnessManager() {
    mDetector = nullptr;
}

void ImageBrightnessManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    AbsVisionManager::init(cfg);
    mDetector->init(cfg);
}

void ImageBrightnessManager::deinit() {
    AbsVisionManager::deinit();
    if (mDetector != nullptr) {
        mDetector->deinit();
        mDetector = nullptr;
    }
}

bool ImageBrightnessManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_IMAGE_BRIGHTNESS);
}

void ImageBrightnessManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_IMAGE_BRIGHTNESS);
    mDetector->doDetect(request, result);
    auto face = result->getFaceResult()->faceInfos[0];
    VLOGD(TAG, "source[%d] brightnessSingle[%d]", mRtConfig->sourceId, face->stateBrightnessSingle);
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("ImageBrightnessManager", ABILITY_IMAGE_BRIGHTNESS,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<ImageBrightnessManager>());
});

}