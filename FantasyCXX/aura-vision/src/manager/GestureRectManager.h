#ifndef VISION_GESTURE_MANAGER_H
#define VISION_GESTURE_MANAGER_H

#include <deque>
#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/GestureRequest.h"
#include "vision/core/result/GestureResult.h"

namespace aura::vision {
class GestureRectDetector;

/**
 * @brief 手势管理器
 * */
class GestureRectManager : public AbsVisionManager {

public:
    GestureRectManager();
    ~GestureRectManager() override;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;
    void doDetect(VisionRequest *request, VisionResult *result) override;

    std::vector<std::deque<int>> _hand_lists;
    std::shared_ptr<GestureRectDetector> _detector;
};

} // namespace vision

#endif //VISION_GESTURE_MANAGER_H
