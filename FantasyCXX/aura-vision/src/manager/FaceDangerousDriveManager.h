
#pragma once

#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "vision/util/ObjectPool.hpp"
#include "detector/FaceDangerousDriveDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief 危险驾驶策略处理
 * */
class FaceDangerousDriveStrategy : public AbsVisionStrategy<FaceInfo>,
                                   public ObjectPool<FaceDangerousDriveStrategy> {
public:
    explicit FaceDangerousDriveStrategy(RtConfig *cfg);

    ~FaceDangerousDriveStrategy() override;

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
    const bool kFaceDangerAngleLimitSwitch = true; // 是否开启角度限制

    bool _record_state; // 记录关键点数据的状态，true-已经记录了，false-没有几录
    /** 记录的最后一帧的关键点数据 */
    VPoint lastLandmark2D106[106]; // 记录的最后一帧的关键点数据
    /** 抽烟滑窗 */
    CustomSateDutyFactorWindow smokeWindow;
    /** 喝水滑窗 */
    CustomSateDutyFactorWindow drinkWindow;
    /** 比嘘滑窗 */
    CustomSateDutyFactorWindow silenceWindow;
    /** 张嘴滑窗 */
    CustomSateDutyFactorWindow openMouthWindow;
    /** 口罩滑窗 */
    CustomSateDutyFactorWindow maskCoverWindow;
    /** 捂嘴滑窗 */
    CustomSateDutyFactorWindow coverMouthWindow;
    /** 普通滑窗：香烟是否燃烧的滑窗 */
    MultiValueSlidingWindow smokeBurningWindow;
};

/**
 * @brief 人脸危险驾驶管理器
 * */
class FaceDangerousDriveManager : public AbsVisionManager {
public:
    FaceDangerousDriveManager();

    ~FaceDangerousDriveManager() override;

    void clear() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

    void onConfigUpdated(int key, float value) override;

    void init(RtConfig *cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceDangerousDriveDetector> detector;

    std::map<int, FaceDangerousDriveStrategy *> dangerDriveStrategyMap;
};

} // namespace vision
