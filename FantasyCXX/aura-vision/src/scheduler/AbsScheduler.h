#ifndef VISION_ABS_SCHEDULER_H
#define VISION_ABS_SCHEDULER_H

#include <memory>
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/util/log.h"

#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

class AbsScheduler {
public:
    AbsScheduler() = default;

    virtual ~AbsScheduler() = default;

    virtual void run(VisionRequest *request, VisionResult *result) = 0;

    /**
     * 注册初始化所有Manager
     */
    void registerManagers();

    /**
     * 注入ManagerRegistry,为初始化所有Manager
     * @param registry
     */
    virtual void injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry> &registry);

    /**
     * 初始化所有Manager
     * @param cfg
     */
    virtual void initManagers(RtConfig *cfg);

    /**
     * 反初始化所有z资源
     */
    virtual void deinit();

public:
    /**
     * 执行对应能力的检测
     * @param id  AbilityId
     * @param request VisionRequest
     * @param result VisionResult
     */
    void runDetection(AbilityId id, VisionRequest *request, VisionResult *result);

    /**
     * 判断前置条件确定是否执行对应能力的检测
     * @param id  AbilityId
     * @param request VisionRequest
     * @param result VisionResult
     */
    void runDetectionIF(AbilityId id, bool cond, VisionRequest *request, VisionResult *result);

    /**
     * 执行指定能力的检测。此处的AbilityId一般由上层的VisionRequest传入直接检测
     * @param id  AbilityId
     * @param request VisionRequest
     * @param result VisionResult
     */
    void runDetectionForcibly(AbilityId id, VisionRequest *request, VisionResult *result);

    void runDetectionForciblyIF(AbilityId id, bool cond, VisionRequest *request, VisionResult *result);

    void runDetectionAndForciblyRely(AbilityId id, AbilityId rely_id, VisionRequest *request, VisionResult *result);

    bool checkFunctionSwitchAndState(AbilityId id, bool state);

    std::shared_ptr<VisionManagerRegistry> getManagerRegistry();

protected:
    std::shared_ptr<VisionManagerRegistry> pMgrRegistry = nullptr;
    RtConfig *mRtConfig = nullptr;
};

enum SchedulerType {
    SCHED_NAIVE,
    SCHED_DAG
};

template<SchedulerType type>
inline std::shared_ptr<AbsScheduler> make_scheduler() {
    VLOGE("SchedulerFactory", "scheduler %d unsupported!", type);
    return nullptr;
}

} // namespace vision


#endif //VISION_ABS_SCHEDULER_H
