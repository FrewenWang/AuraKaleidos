#ifndef VISION_FACE_MOUTH_LANDMARK_MANAGER_H
#define VISION_FACE_MOUTH_LANDMARK_MANAGER_H

#include <memory>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "detector/FaceMouthLandmarkDetector.h"

namespace aura::vision {

/**
 * @brief 人脸关键点管理器
 * */
class FaceMouthLandmarkManager : public AbsVisionManager {
public:
    FaceMouthLandmarkManager();

    ~FaceMouthLandmarkManager() override = default;

    void clear() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceMouthLandmarkDetector> detector;
};

} // namespace vision

#endif //VISION_FACE_MOUTH_LANDMARK_MANAGER_H
