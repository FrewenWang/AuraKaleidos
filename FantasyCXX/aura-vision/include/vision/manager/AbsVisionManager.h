//
// Created by Li,Wendong on 2018/12/24.
//

#ifndef VISION_ABS_VISION_MANAGER_H
#define VISION_ABS_VISION_MANAGER_H

#include <scheduler/dag/Node.h>
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/core/common/VFrame.h"
#include "vision/config/runtime_config/RtConfig.h"

namespace aura::vision {
/**
 * @brief get the ability detected status
 */
#define VA_GET_DETECTED(ability) result->isAbilityExec(ability)
#define VA_CHECK_DETECTED(ability) return !result->isAbilityExec(ability)
#define VA_SET_DETECTED(ability)                    \
do {                                                \
    result->setAbilityExec(ability);              \
} while (0)

class RtConfig;

class ConfigListener {
public:
    virtual void onConfigUpdated(int key, float value) = 0;
};

class CmdListener {
public:
    virtual bool onAbilityCmd(int cmd) = 0;
};

class DetectResultListener {
public:
    virtual void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face = nullptr) = 0;

    virtual void onNoGesture(VisionRequest *request, VisionResult *result, GestureInfo *gesture = nullptr) = 0;

    virtual void onNoBody(VisionRequest *request, VisionResult *result, BodyInfo *body = nullptr) = 0;

    virtual void onNoLiving(VisionRequest *request, VisionResult *result, LivingInfo *living = nullptr) = 0;
};

class Node;

class AbsVisionManager : public ConfigListener, public CmdListener, public DetectResultListener, public Node {
public:
    virtual ~AbsVisionManager() = default;

    /**
     * 各能力检测入口
     * @param request 检测请求
     * @param result 检测结果
     */
    virtual void detect(VisionRequest *request, VisionResult *result);

    /**
     * @brief 状态清除
     */
    virtual void clear();

    /**
     * @brief 清除触发次数
     */
    virtual void clearTriggerAccumulative() { };

    /**
     * 未检测到人脸执行的策略
     * @param result
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face = nullptr) override { };

    /**
     * 未检测到手势执行的策略
     * @param result
     */
    void onNoGesture(VisionRequest *request, VisionResult *result, GestureInfo *gesture = nullptr) override { };

    /**
     * 未检测到肢体的策略
     * @param request
     * @param result
     * @param body
     */
    void onNoBody(VisionRequest *request, VisionResult *result, BodyInfo *body = nullptr) override { };

    /**
     * 未检测到活体的策略处理
     * @param request
     * @param result
     * @param living
     */
    void onNoLiving(VisionRequest *request, VisionResult *result, LivingInfo *living = nullptr) override { };

    /**
     * 运行时的配置更新操作
     * @param key 配置参数的键
     * @param value 配置参数的值
     */
    void onConfigUpdated(int key, float value) override { };

    /**
     * 运行时的配置更新操作
     * @param key 配置参数的键
     * @param value 配置参数的值
     */
    bool onAbilityCmd(int cmd) override {
        return false;
    };

    /**
     * 初始化数据
     */
    virtual void init(RtConfig *cfg) {
        mRtConfig = cfg;
    };

    /**
     * 反初始化数据
     */
    virtual void deinit() {
        clear();
        mRtConfig = nullptr;
    };

    /**
      * 设置不同帧率下动态滑窗长度和占空比，每个manager自定义滑窗长度因子和占空比因子：
      *     滑窗长度因子：实时帧率下，对应滑窗长度与该实时帧率的比值
      *     占空比因子：实时帧率下，对应滑窗的占空比
      */
    virtual void setupSlidingWindow() {};

    std::string name;
    int index;

    /** 动态滑窗实时帧率的上限 */
    static constexpr int WINDOW_UPPER_FPS = 25;
    /** 动态滑窗实时帧率的下限 */
    static constexpr int WINDOW_LOWER_FPS = 3;

    /** 默认的原子能力多帧滑窗长度 */
    static constexpr float DEFAULT_WINDOW_LENGTH = 15;
    /** 默认的原子能力多帧滑窗结果触发占空比 */
    static constexpr float DEFAULT_TRIGGER_DUTY_FACTOR = 0.8;
    /** 默认的原子能力多帧滑窗结果结束占空比 */
    static constexpr float DEFAULT_END_DUTY_FACTOR = 0.4;

    /** 实时帧率下，默认使用的滑窗占空比 */
    static constexpr float DEFAULT_W_DUTY_FACTOR = 0.8;
    /** 实时帧率下，默认使用的对应滑窗长度与实时帧率的比值 */
    static constexpr float DEFAULT_W_LENGTH_RATIO_1_0 = 1.0;
    static constexpr float DEFAULT_W_LENGTH_RATIO_1_5 = 1.5;
    static constexpr float DEFAULT_W_LENGTH_RATIO_2_0 = 2.0;
    static constexpr float DEFAULT_W_LENGTH_RATIO_3_0 = 3.0;

private:
    /**
     * @param request
     * @param result
     * @return 是否继续执行 doDetect()
     */
    virtual bool preDetect(VisionRequest *request, VisionResult *result) {
        return true;
    };

    /**
     * 具体地检测操作
     * @param request
     * @param result
     */
    virtual void doDetect(VisionRequest *request, VisionResult *result) { };

protected:
    RtConfig *mRtConfig;
    /** 是否强制进行人脸检测的标志变量 */
    bool forceDetectCurFrame = true;
    /** 上一帧检测的时间戳。默认起始值为0 */
    std::int64_t preDetectMillis = 0;
    /** 自定义固定帧率下的单帧执行时长。比如3：333、4：250、5：200、6：167 */
    const static int FIXED_DETECT_DURATION = 200;

    /**
     * @deprecated  TODO 暂时不要启用，需要考虑问题比较多。
     * 根据自定义的固定帧率判断当前帧率是否检测
     * @param detectDuration  自定义固定帧率下的单帧执行时长。比如3：333、4：250、5：200、6：
     * @param forceDetect 是否强制进行检测（业务侧根据自己需求判断是否需要进行检测：比如一旦检测到有效结果，则每帧强制检测）
     * @return  true  则表示当前帧率需要执行检测  false 则表示当前帧不需要检测
     */
    bool checkFpsFixedDetect(int detectDuration, const bool &forceDetect);

    bool execute() override;
};

} // namespace vision

#endif // VISION_ABS_VISION_MANAGER_H
