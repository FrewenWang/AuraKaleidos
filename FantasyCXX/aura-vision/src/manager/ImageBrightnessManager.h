
#pragma once

#include "vision/manager/AbsVisionManager.h"
#include "detector/ImageBrightnessDetector.h"

namespace aura::vision {

class ImageBrightnessManager : public AbsVisionManager {

public:
    ImageBrightnessManager();

    ~ImageBrightnessManager() override;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    RtConfig *mRtConfig;
    /** 图像亮度检测器Detector */
    std::shared_ptr<ImageBrightnessDetector> mDetector;

};

}
