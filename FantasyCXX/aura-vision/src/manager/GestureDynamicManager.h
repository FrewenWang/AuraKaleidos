
#ifndef VISION_GESTURE_DYNAMIC_MANAGER_H
#define VISION_GESTURE_DYNAMIC_MANAGER_H

#include <deque>
#include <vector>
#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/GestureRequest.h"
#include "vision/core/result/GestureResult.h"
#include "util/sliding_window.h"

namespace vision {
/**
 * @brief 动态手势管理器
 * */
class GestureDynamicManager : public AbsVisionManager {
public:
    GestureDynamicManager();

    ~GestureDynamicManager() override = default;

    void clear() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

private:
    void setup_sliding_window();

    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    /**
     * 检测是否触发某一动态手势
     * @param detectDynamicType         要检测的动态手势类型
     * @param triggerStartType          标示动态手势开始的手势类型
     * @param triggerEndType            标示动态手势结束的手势类型
     * @param curStaticGestureType      当前帧检测到的静态手势类型
     * @param triggerStartTime          动态手势开始的记录时间
     * @param outputType                检测后输出的最终动态手势类型
     * @return
     */
    bool by_dynamic_type_detect(int detectDynamicType, int triggerStartType, int triggerEndType,
                                int curStaticGestureType, std::int64_t &triggerStartTime, int &outputType);

    // 当前时间
    std::int64_t nowTime;
    // 当前帧检测到的静态手势类型(不包含挥手相关的静态手势类型)
    std::int32_t curStaticGestureType;
    // 当前帧检测到的与挥手相关的静态手势类型
    std::int32_t curStaticGestureWaveType;
    // 检测到最后一帧有效动态手势的时间
    std::int64_t lastTriggerDynamicTime;
    // 动作捏开始的时间
    std::int64_t startPinchTime;
    // 动作抓开始的时间
    std::int64_t startGraspTime;
    // 动作左挥手开始的时间
    std::int64_t startLeftWaveTime;
    // 动作右挥手开始的时间
    std::int64_t startRightWaveTime;

    // 检测到的第二个动态手势与第一个间隔1s才有效
    const int WATI_TIME_LIMIT = 1000; // ms
    // 每个动态手势必须在1s内触发完毕才有效
    const int DETECT_TIME_LIMIT = 1000; // ms
    // 默认静态滑窗长度
    const int DEFAULT_WINDOW_LEN = 1;
    // 默认静态滑窗占空比
    const float DEFAULT_DUTY_FACTOR = 1.0f;
    // 默认动态滑窗长度
    const int DEFAULT_DYNAMIC_WINDOW_LEN = 5;
    // 默认动态滑窗占空比
    const float DEFAULT_DYNAMIC_DUTY_FACTOR = 0.6f;

    // 算法模型输出手势类型与实际定义手势类型映射
    std::unordered_map<std::int32_t, std::int16_t> gestMap;

    // 针对静态手势的通用静态滑窗
    MultiValueSlidingWindow gestureStaticWindow;
    // 针对挥手静态手势相关操作静态滑窗（目前挥手相关的单独操作，待模型优化后再将挥手手势与其他动态手势合并统一处理）
    MultiValueSlidingWindow waveStaticWindow;
    // 针对动态手势的动态滑窗
    MultiValueSlidingWindow gestureDynamicWindow;
};
} // namespace vision
#endif //VISION_GESTURE_DYNAMIC_MANAGER_H