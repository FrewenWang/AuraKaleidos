

#ifndef VISION_BODY_LANDMARK_MANAGER_H
#define VISION_BODY_LANDMARK_MANAGER_H

#include <memory>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/BodyRequest.h"
#include "vision/core/result/BodyResult.h"
#include "detector/BodyLandmarkDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief Body关键点管理器
 * */
class BodyLandmarkManager : public AbsVisionManager {
public:
    BodyLandmarkManager();

    ~BodyLandmarkManager() override = default;

    void clear() override;

    void onNoBody(VisionRequest* request, VisionResult* result,BodyInfo *body) override;

    void init(RtConfig* cfg) override;

    void deinit() override;

    void setupSlidingWindow() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<BodyLandmarkDetector> bodyDetector;
    CustomSateDutyFactorWindow noBodyWindow;
};

} // namespace aura::vision

#endif //VISION_BODY_LANDMARK_MANAGER_H
