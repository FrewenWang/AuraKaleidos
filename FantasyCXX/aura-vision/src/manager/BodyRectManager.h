#pragma once

#include <deque>
#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/BodyRequest.h"
#include "vision/core/result/BodyResult.h"
#include "AbsVisionStrategy.h"

namespace aura::vision {

enum FaceBodyStatus : short {
    STATUS_UNKNOWN = -1,          // -1 未知默认状态
    STATUS_NO_FACE_BODY = 0,      // 0 无人脸无肢体
    STATUS_HAS_FACE_NO_BODY = 1,  // 1 有人脸无肢体
    STATUS_NO_FACE_HAS_BODY = 2,  // 2 无人脸有肢体
    STATUS_HAS_FACE_HAS_BODY = 3, // 3 有人脸有肢体
};

class BodyRectDetector;

/**
 * @brief
 * */
class BodyRectStrategy : public AbsVisionStrategy<BodyInfo>, public ObjectPool<BodyRectStrategy> {
public:
    explicit BodyRectStrategy(RtConfig *cfg);

    ~BodyRectStrategy() override;

    /**
     * 根据单帧数据执行逻辑处理
     * @param body框检测模型输出信息
     */
    void execute(BodyInfo *body) override;

    /**
     * 执行Body框策略
     * @param pFaceInfo
     * @param pBodyInfo
     */
    void execute(VisionRequest *request, FaceInfo *pFaceInfo, BodyInfo *pBodyInfo, int statusCode);
    /**
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    void predictBodyLocation(VPoint &leftUp, VPoint &rightDown, VPoint &rectCenter, int frameWidth, int frameHeight,
                             float a, float b, float c, float d, float e);

    void calculateRectBox(VPoint &leftUp, VPoint &rightDown, VPoint &rectCenter, float a, float b, float c, float d,
                          float fedge);

    const float angleMax = 30.0f;
    const float angleMin = -30.0f;
    const float widthDeltaMaxA = 450.0f;
    const float widthDeltaMaxB = 600.0f;
    const float widthHeightRatio = 1.35f;
    float widthDelta = 0.0f;
    float heightDelta = 0.0f;
    float shoulderRatio = 0.0f;
    float fedge = 0.0f;
};

/**
 * @brief 肢体框检测管理器(现在使用的头肩框检测模型)
 * */
class BodyRectManager : public AbsVisionManager {

public:
    BodyRectManager();

    ~BodyRectManager() override = default;

    void init(RtConfig *cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    std::shared_ptr<BodyRectStrategy> bodyRectStrategy = nullptr;

    std::shared_ptr<BodyRectDetector> detector;
    /**
     * 计算人脸框和肢体框的IOU分值
     * @param pFaceInfo
     * @param pBodyInfo
     * @return
     */
    float getMatchIOU(FaceInfo *pFaceInfo, BodyInfo *pBodyInfo);
    /**
     * 根据肢体框和人脸框计算出来的临时匹配的框区域
     */
    VPoint tmpMatchLT;
    VPoint tmpMatchRB;
    /**
     * 人脸框和肢体框的IOU匹配阈值
     */
    const float iouThreshold = 0.5f;
    /** 进行储存推理最终肢体框的ID的集合 **/
    std::vector<int> useFaceRectList;                      // 使用人脸框进行肢体框计算的索引集合
    std::vector<int> useBodyRectList;                      // 使用肢体框进行肢体框计算的索引集合
    // 使用肢体框和人脸框融合进行肢体框计算的索引匹配集合,注意first为肢体、second为人脸
    std::vector<std::pair<int, int>> bodyFaceBothRectList;
    /**
     * 标志检测出来的肢体框的数量
     */
    int bodyCount;

    std::vector<BodyInfo *> curOrderedBody{};
};

} // namespace aura::vision
