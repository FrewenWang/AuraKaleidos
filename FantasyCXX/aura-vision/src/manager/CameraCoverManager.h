#pragma once

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "util/sliding_window.h"
#include "detector/Source1CameraCoverDetector.h"
#include "detector/Source2CameraCoverDetector.h"
#include <tuple>

namespace aura::vision {
/**
 * @brief 摄像头遮挡管理器
 * */
class CameraCoverManager : public AbsVisionManager {
public:
    CameraCoverManager();

    ~CameraCoverManager() override = default;

    void init(RtConfig *cfg) override;

    void deinit() override;

    void clear() override;

    void setupSlidingWindow() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    /** 摄像头遮挡的滑窗策略逻辑 */
    CustomSateDutyFactorWindow cameraCoverWindow;
    /**
     * 摄像头遮挡不同的遮挡检测器Detector
     */
    std::shared_ptr<AbsCameraCoverDetector> detector;
    /**  VisSourceConfig里面设置的遮挡阈值 */
    float cameraCoverThreshold = 0.f;
};

} // namespace aura::vision