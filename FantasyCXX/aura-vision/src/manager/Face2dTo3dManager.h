

#ifndef VISION_FACE_2DTO3D_MANAGER_H
#define VISION_FACE_2DTO3D_MANAGER_H

#include <vector>

#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "vision/manager/AbsVisionManager.h"

#include "detector/Face2dTo3dDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief 人脸视线追踪管理器
 * */
class Face2dto3dManager : public AbsVisionManager {
public:
    Face2dto3dManager();

    ~Face2dto3dManager() override;

    void clear() override;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    void setup_sliding_window();

    bool is_init_model;

private:
    std::shared_ptr<Face2Dto3DDetector> _detector;
};

} // namespace aura::vision

#endif //VISION_FACE_2DTO3D_MANAGER_H
