
#pragma once

#include <map>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "detector/FaceCallDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief
 * */
class FaceCallStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceCallStrategy> {
public:
    explicit FaceCallStrategy(RtConfig *cfg);

    ~FaceCallStrategy();

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
     * 根据 Config 字段更新相关属性
     * @param key Config的字段标识
     * @param value 更新的数值
     */
    void onConfigUpdated(int key, float value) override;

    /**
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    /** 判断左右耳朵单帧打电话识别的滑窗 */
    CustomSateDutyFactorWindow faceCallWindow;
    /** 判断耳朵边有遮挡物的滑窗，此滑窗无意义，后续可去除 */
    CustomSateDutyFactorWindow gestureNearbyWindow;
};

class FaceCallManager : public AbsVisionManager {
public:
    FaceCallManager();

    ~FaceCallManager() override;

    /**
     * 初始化FaceCallManager。主要是初始化对应的Detector
     * @param cfg
     */
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

    /**
     * SourceConfig更新时结果回调。支持业务实时更新config配置
     * @param key  Key
     * @param value Value
     */
    void onConfigUpdated(int key, float value) override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    std::shared_ptr<FaceCallDetector> detector;

    std::map<int, FaceCallStrategy *> faceCallStrategyMap;
};

} // namespace aura::vision

