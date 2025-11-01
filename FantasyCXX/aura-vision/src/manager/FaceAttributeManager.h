
#ifndef VISION_FACE_ATTRIBUTE_MANAGER_H
#define VISION_FACE_ATTRIBUTE_MANAGER_H

#include <map>

#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/manager/AbsVisionManager.h"

#include "detector/FaceAttributeDetector.h"
#include "manager/AbsVisionStrategy.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief 属性策略处理
 */
class FaceAttributeStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceAttributeStrategy> {
public:
    explicit FaceAttributeStrategy(RtConfig* cfg);

    ~FaceAttributeStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo* face) override;

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

    void setupSlidingWindow() override;

private:
    MultiValueSlidingWindow _age_window;
    MultiValueSlidingWindow _gender_window;
    MultiValueSlidingWindow _race_window;
    MultiValueSlidingWindow _glass_window;
};

class FaceAttributeManager : public AbsVisionManager {
public:
    FaceAttributeManager();

    ~FaceAttributeManager() override;

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
    std::shared_ptr<FaceAttributeDetector> detector;
    std::map<int, FaceAttributeStrategy*> attributeStrategyMap;
};
}// namespace aura::vision

#endif //VISION_FACE_ATTRIBUTE_MANAGER_H
