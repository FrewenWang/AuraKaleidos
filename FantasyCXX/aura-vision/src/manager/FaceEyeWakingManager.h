

#ifndef VISION_FACE_EYE_WAKING_MANAGER_H
#define VISION_FACE_EYE_WAKING_MANAGER_H

#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "util/sliding_window.h"

namespace aura::vision {
/**
 * @brief 人脸特征值管理器
 * */
class FaceEyeWakingManager : public AbsVisionManager {
public:
    FaceEyeWakingManager();

    ~FaceEyeWakingManager() override;

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

    void setupSlidingWindow() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    bool detect_eye_angle(FaceInfo *fi);

    void detect_face_angle(FaceInfo *fi);

    FaceInfo *_fi;

    short _eye_direction;
    short _face_direction;
    float _is_eye_wake;
    float _left_eye_dis;
    float _right_eye_dis;
    float _eye_radius;
    float _eye_dis;
    float _yaw;

    const char _k_waking_eye_forward = 0;
    const char _k_waking_eye_left = 1;
    const char _k_waking_eye_right = 2;
    const char _k_waking_face_forward = 0;
    const char _k_waking_face_left = 1;
    const char _k_waking_face_right = 2;

    CustomSateDutyFactorWindow _wake_window;
};

} // namespace vision

#endif // VISION_FACE_EYE_WAKING_MANAGER_H
