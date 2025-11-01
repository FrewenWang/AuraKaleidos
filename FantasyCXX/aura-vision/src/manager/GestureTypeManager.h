#ifndef VISION_GESTURE_TYPE_MANAGER_H
#define VISION_GESTURE_TYPE_MANAGER_H

#include <deque>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/GestureRequest.h"
#include "vision/core/result/GestureResult.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief 手势种类
 * */
class GestureTypeManager : public AbsVisionManager {
public:
    GestureTypeManager();

    ~GestureTypeManager() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

    void onNoGesture(VisionRequest *request, VisionResult *result, GestureInfo *gesture) override;

    void clear() override;

    void setupSlidingWindow() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    std::unordered_map<int, short> _gest_map;

    MultiValueSlidingWindow gestureWindow;
};

} // namespace vision

#endif //VISION_GESTURE_MANAGER_H
