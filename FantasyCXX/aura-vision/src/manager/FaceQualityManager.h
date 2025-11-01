

#ifndef VISION_FACE_QUALITY_MANAGER_H
#define VISION_FACE_QUALITY_MANAGER_H

#include <memory>
#include <deque>
#include <map>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "detector/FaceQualityDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief
 * */
class FaceQualityStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceQualityStrategy> {
public:
    explicit FaceQualityStrategy(RtConfig *cfg);

    ~FaceQualityStrategy();

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
    //滑窗配置
    CustomSateDutyFactorWindow faceCoverWindow;
    CustomSateDutyFactorWindow leftEyeCoverWindow;
    CustomSateDutyFactorWindow rightEyeCoverWindow;
    CustomSateDutyFactorWindow mouthCoverWindow;
    CustomSateDutyFactorWindow blurWindow;
    CustomSateDutyFactorWindow noiseWindow;
};

/**
 * @brief 图像质量管理器
 * */
class FaceQualityManager : public AbsVisionManager {
public:
    FaceQualityManager();

    ~FaceQualityManager() override;

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
    std::shared_ptr<FaceQualityDetector> detector;

    std::map<int, FaceQualityStrategy *> faceQualityMap;
};

} // namespace vision

#endif //VISION_FACE_QUALITY_MANAGER_H
