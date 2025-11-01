#ifndef VISION_LIP_MOVEMENT_MANAGER_H
#define VISION_LIP_MOVEMENT_MANAGER_H

#include <vector>

#include "AbsVisionStrategy.h"
#include "util/sliding_window.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/manager/AbsVisionManager.h"

namespace aura::vision {
/**
 * @brief
 * */
class FaceLipMovementStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceLipMovementStrategy> {
public:
    explicit FaceLipMovementStrategy(RtConfig *cfg);

    ~FaceLipMovementStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo *face) override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, FaceInfo *face) override;

    /**
     * @brief 初始化
     */
    void init();

    /**
     * 清空数据
     */
    void clear() override;

private:
    /**
     * @brief 重置所有数据
     */
    void resetAllData();

    /**
     * 当检测到嘴部遮挡、头姿超过边界，且时间超时一个时间窗口时，从新开始检测唇动
     */
    void checkResetState(int windowTime);

private:
    /** 刚开始检测的第一帧，或者丢失人脸后重新开始检测的第一帧不进行唇动判断，lastLipDistance默认-1，当第二帧来到才进行第一次唇动判断 */
    const double DEFAULT_LAST_LIP_DISTANCE = -1.0f;
    /** 最近一次变化时间戳 */
    int64_t lastChangeTime = 0;
    /** 最近一次嘴部张开幅度，默认为defaultLastLipDistance */
    double lastLipDistance = DEFAULT_LAST_LIP_DISTANCE;
    /** 当前的单帧结果 */
    bool lastSingleLip = false;
    /** 前后两帧的唇动变化量 */
    float change = 0.0f;
    float postChange = 0.0f;
    /** 前后两帧唇动距离是否超过唇动阈值 */
    bool singleLip = false;
    /** 当前帧时刻时间(毫秒) */
    int64_t curTime = 0;
    /** 当前帧时间与起始帧时间的间隔(毫秒) */
    int64_t durTime = 0;
    /** 唇动阈值 */
    float lipThreshold = 10.0f;

    struct LipValues {
        LipValues() {

        }
        LipValues(float d, float y, float p) {
            lipDistance = d;
            yaw = y;
            pitch = p;
        }
        float lipDistance = 0;
        float yaw = 0;
        float pitch = 0;
        std::string toString() {
            return "[" + std::to_string(lipDistance) + ", " + std::to_string(yaw) + ", " + std::to_string(pitch) + "]";
        }
    };

    std::deque<LipValues> lipDeque;
    short windowLen = 10;
    short lipDequeLen = 0;
    LipValues maxLip;
    LipValues minLip;
};

class FaceLipMovementManager : public AbsVisionManager {
public:
    FaceLipMovementManager();

    ~FaceLipMovementManager() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

    void clear() override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::map<int, FaceLipMovementStrategy *> faceLipMovementMap;
};
} // namespace vision

#endif // VISION_LIP_MOVEMENT_MANAGER_H
