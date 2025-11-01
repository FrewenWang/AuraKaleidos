#ifndef VISION_FACE_DETECTION_MANAGER_H
#define VISION_FACE_DETECTION_MANAGER_H

#include <map>
#include <memory>

#include "AbsScheduler.h"
#include "opencv2/core/core.hpp"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "util/FaceTrackHelper.h"

namespace aura::vision {

/**
 * @brief 用于人脸跟踪的模板
 */
class FaceTemplate {
public:
    FaceTemplate(int id, bool is_template);

    FaceTemplate(int id, bool is_template, float l, float t, float r, float b);

    /**
     * 更新人脸框数据
     * @param rect
     */
    void updateRect(VRect &&rect);

    bool iou(const VRect &input);

public:
    /** 模板ID */
    int id;
    /** 是否是模板的标志变量 */
    bool isTemplate;
    /** 人脸框数据 */
    VRect vRect;
    /** 模板信息数据 */
    cv::Mat templateMat;
};

/**
 * @brief 多人脸管理器
 * */
class FaceTrackSubScheduler : public AbsScheduler {
public:
    ~FaceTrackSubScheduler() override = default;

    void run(VisionRequest *request, VisionResult *result) override;

    void set_config(RtConfig *cfg);

private:
    /**
     * 使用模型检测
     * @param request
     * @param result
     */
    void modelDetect(VisionRequest *request, VisionResult *result);

    /**
     * 人脸跟随（目前采用人脸跟踪策略）
     * @param request
     * @param result
     */
    void faceTrackDetect(VisionRequest *request, VisionResult *result);

    /**
     * 模板匹配（目前采用OpenCV模板匹配检测策略）
     * @param request
     * @param result
     */
    void matchTemplate(VisionRequest *request, VisionResult *result);

    /**
     * 初始化人脸跟踪
     */
    void initFaceTrack();

private:
    /**
     * 人脸模板集合
     */
    std::map<int, FaceTemplate> faceTemplateList;
    short detectMethod = FaceDetectMethod::DETECT;

    // 人脸跟踪处理对象
    FaceTrackHelper mFaceTrackHelper;
};

} // namespace vision

#endif //VISION_FACE_DETECTION_MANAGER_H
