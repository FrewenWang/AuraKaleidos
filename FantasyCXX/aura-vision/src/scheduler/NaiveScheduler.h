#ifndef VISION_NAIVE_SCHEDULER_H
#define VISION_NAIVE_SCHEDULER_H

#include "AbsScheduler.h"
#include "FaceTrackSubScheduler.h"

#include <memory>

namespace aura::vision {

class FaceTrackSubScheduler;

class NaiveScheduler : public AbsScheduler {
public:
    NaiveScheduler();

    ~NaiveScheduler() override = default;

    void run(VisionRequest *request, VisionResult *result) override;

    void injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry> &registry) override;

    void initManagers(RtConfig *cfg) override;

private:
    /**
     * @brief 执行人脸信息相关的检测能力
     * @param request  VisionRequest
     * @param result    VisionResult
     */
    void run_face_detections(VisionRequest *request, VisionResult *result);

    /**
     * @brief 执行手势相关的检测能力
     * @param request  VisionRequest
     * @param result    VisionResult
     */
    void run_gesture_detections(VisionRequest *request, VisionResult *result);

    /**
     * 执行猫狗宠物婴儿活体检测
     * @param request  VisionRequest
     * @param result    VisionResult
     */
    void runLivingDetections(VisionRequest *request, VisionResult *result);

    /**
    * @brief 执行身体相关的检测能力
    * @param request  VisionRequest
    * @param result    VisionResult
    */
    void runBodyDetections(VisionRequest *request, VisionResult *result);

private:
    /** 人脸跟踪逻辑的调度器 */
    std::unique_ptr<FaceTrackSubScheduler> faceTrackSubScheduler;
};

template<>
inline std::shared_ptr<AbsScheduler> make_scheduler<SCHED_NAIVE>() {
    return std::dynamic_pointer_cast<AbsScheduler>(std::make_shared<NaiveScheduler>());
}

}  // namespace vision

#endif //VISION_NAIVE_SCHEDULER_H
