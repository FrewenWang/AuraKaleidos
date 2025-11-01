
#pragma once

#ifdef BUILD_3D_LANDMARK

#include <memory>

#include "util/sliding_window.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "vision/manager/AbsVisionManager.h"
#include "detector/Face3dLandmarkDetector.h"

namespace aura::vision {
/**
 * @brief 人脸3D关键点管理器
 * */
class Face3dLandmarkManager : public AbsVisionManager {
public:
    Face3dLandmarkManager();

    ~Face3dLandmarkManager() override = default;

    void clear() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;
    std::shared_ptr<Face3dLandmarkDetector> _detector;
    /**
     * 能力层Manager接收到指令消息
     * @param key
     */
    bool onAbilityCmd(int cmd) override;
};

} // namespace aura::vision

#endif
