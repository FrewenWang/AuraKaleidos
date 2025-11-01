#ifndef VISION_GESTURE_LANDMARK_MANAGER_H
#define VISION_GESTURE_LANDMARK_MANAGER_H

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/GestureRequest.h"
#include "vision/core/result/GestureResult.h"
#include "detector/GestureLandmarkDetector.h"

namespace aura::vision {

/**
 * @brief 手势管理器
 * */
class GestureLandmarkManager : public AbsVisionManager {

public:
    GestureLandmarkManager();

    ~GestureLandmarkManager() override = default;

    void init(RtConfig* _cfg) override;

    void deinit() override;

private:

    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<GestureLandmarkDetector> _detector;
};

} // namespace vision

#endif //VISION_GESTURE_MANAGER_H
