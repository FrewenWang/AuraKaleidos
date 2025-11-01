

#ifndef VISION_FACE_RECT_MANAGER_H
#define VISION_FACE_RECT_MANAGER_H

#include <memory>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "detector/FaceRectDetector.h"

namespace aura::vision {

/**
 * @brief 人脸框管理器
 * */
class FaceRectManager : public AbsVisionManager {
public:
    FaceRectManager();
    ~FaceRectManager() override = default;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;
    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceRectDetector> _detector;
};

} // namespace vision

#endif //VISION_FACE_RECT_MANAGER_H
