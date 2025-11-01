
#pragma once

#include <deque>
#include <map>
#include "manager/AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "vision/manager/AbsVisionManager.h"
#include "detector/FaceNoInteractLivingDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief
 * */
class FaceNoInteractLivingStrategy
        : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceNoInteractLivingStrategy> {
public:
    explicit FaceNoInteractLivingStrategy(RtConfig* cfg);

    ~FaceNoInteractLivingStrategy();

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
    void onNoFace(VisionRequest* request, FaceInfo* face) override;

    /**
     * 清空数据
     */
    void clear() override;

private:
    /**
     * 设置滑窗相关属性信息，对象构造的时候调用。
     */
    void setup_sliding_window();

private:
    /**
     * 无感活体自定义系数的滑窗对象
     */
    CustomSateDutyFactorWindow noInteractLivingWindow;
};

/**
 * @brief 人脸静默活体管理器
 * */
class FaceNoInteractLivingManager : public AbsVisionManager {
public:
    FaceNoInteractLivingManager();

    ~FaceNoInteractLivingManager() override;

    void init(RtConfig* cfg) override;

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
    std::shared_ptr<FaceLivenessDetector> detector;

    std::map<int, FaceNoInteractLivingStrategy*> noInteractLivingMap;
};

} // namespace vision
