#ifndef VISION_LIVING_MANAGER_H
#define VISION_LIVING_MANAGER_H

#include <deque>
#include <vector>
#include <map>

#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/manager/AbsVisionManager.h"

#include "detector/LivingDetector.h"
#include "manager/AbsVisionStrategy.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief 属性策略处理
 */
class LivingStrategy : public AbsVisionStrategy<LivingInfo>, public ObjectPool<LivingStrategy> {
public:
    explicit LivingStrategy(RtConfig *cfg);

    ~LivingStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(LivingInfo *face) override;

    //onNoFace(VisionRequest* request, FaceInfo* face) override;

    /**
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    MultiValueSlidingWindow livingCategoryWindow;
};

/**
 * 猫狗婴儿活体检测管理类
 */
class LivingManager : public AbsVisionManager {
public:
    LivingManager();

    ~LivingManager() override;

    void init(RtConfig *cfg) override;

    void deinit() override;

    void clear() override;

private:
    /**
     *  猫狗婴儿活体检测检测预处理
     * @param request
     * @param result
     * @return  检测结果
     */
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    /**
     * 猫狗婴儿活体检测逻辑
     * @param request
     * @param result
     */
    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<LivingRectDetector> _detector;
    /** 活体检测策略处理类 */
    //LivingStrategy *livingStrategy = nullptr;
    /**
     * 活体检测策略处理类的映射对象
     */
    std::map<int, LivingStrategy *> livingStrategyMap;
};


} // namespace vision

#endif
