
#ifndef VISION_FACE_FEATURE_MANAGER_H
#define VISION_FACE_FEATURE_MANAGER_H

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "opencv2/opencv.hpp"
#include "detector/FaceFeatureDetector.h"

namespace aura::vision {

/**
 * @brief 人脸特征值管理器
 * */
class FaceFeatureManager : public AbsVisionManager {
public:
    FaceFeatureManager();

    ~FaceFeatureManager() override = default;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceFeatureDetector> _detector;
};

} // namespace vision

#endif //VISION_FACE_IDENTIFY_MANAGER_H
