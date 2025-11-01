
#pragma once

#include <deque>
#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "detector/FaceEmotionDetector.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "util/sliding_window.h"

namespace aura::vision {
/**
 * @brief 人脸情绪识别的滑窗策略类
 * */
class FaceEmotionStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceEmotionStrategy> {
public:
    explicit FaceEmotionStrategy(RtConfig *cfg);

    ~FaceEmotionStrategy();

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
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    /**
     * 人脸情绪识别的多值滑窗类
     */
    MultiValueSlidingWindow faceEmotionWindow;
};

/**
 * @brief 表情管理器
 * */
class FaceEmotionManager : public AbsVisionManager {
public:
    FaceEmotionManager();

    ~FaceEmotionManager() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

    void clear() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;


private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceEmotionDetector> detector;

    std::map<int, FaceEmotionStrategy *> emotionStrategyMap;
};

} // namespace aura::vision

