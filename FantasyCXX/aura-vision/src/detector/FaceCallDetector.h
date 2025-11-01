#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vacv/cv.h"

namespace aura::vision {

/**
 * 人脸打电话检测
 */
class FaceCallDetector : public AbsFaceDetector {
public:
    FaceCallDetector();

    ~FaceCallDetector() override;

    int init(RtConfig *cfg) override;

    /**
     * 执行人脸打电话检测
     * @param request
     * @param result
     * @return
     */
    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();
    /**
     * 打电话的模型推理
     * @param request
     * @param result
     * @param face
     * @return
     */
    int doDetectInner(VisionRequest *request, VisionResult *result, FaceInfo **face);
    /** 后处理归一化数组的大小 */
    static const int OUTPUT_DATA_SIZE = 3;
    /** 模型输入要求的宽高尺寸 */
    int inputWidth = 0;
    int inputHeight = 0;
    /** 仿射变换参照图需要裁剪的宽高尺寸 */
    int cropWidth = 0;
    int cropHeight = 0;
    /** 默认第一帧开始检测左耳打电话，后续每一帧交替检测左右耳 */
    bool defaultDetectLeft = true;
    /** 标记是否是检测左耳朵(用于标记是检测左耳打电话，还是检测右耳打电话) */
    bool detectLeftEar = true;
    /** 记录每帧打电话的结果输出分值 */
    float scoreFaceCall = 0.f;
    /** 记录每帧Nearby的结果输出分值 */
    float scoreGestureNearby = 0.f;

    /** 仿射变换 */
    VPoint refLmkRightStart;
    VPoint refLmkLeftStart;
    VPoint *faceCallRefLmkLeft;
    VPoint *faceCallRefLmkRight;

    /** 前后处理中固定系数配置 */
    const float cropWidthRatio = 1.62f;
    const float cropHeightRatio = 1.2f;
    const float cropBiasRatio = 0.1f;
    const float cropBoxBias = 0.5f;
    // 最新阈值：DMSCall230908V14MainNoQATFP16 阈值修改为0.6
    // 最新阈值：DMSCall231123V14MainNoQATFP16 阈值修改为0.8
    const float callScoreNormalAngle = 0.8f;
    //lower call confidence in large angle
    const float callScoreLargeAngle = 0.3f;
    //useStrategy if angle out of setting, easy to confuse gestureNearby and call in large angle,
    //calibrate callScore and gestureNearbyScore in large angle according to attachments in cases
    bool useStrategy = false;
    //calibrate angle according to attachments in cases
    const float leftYawUpper = 10.f;
    const float rightYawUpper = -10.f;

    /** 前处理裁剪的区域框 */
    const vision::VRect cropRect;
    /** 输出的归一化数组，取最大元素下标  */
    float softmaxCallOutput[OUTPUT_DATA_SIZE];
    /**
     * 打电话检测的Index,默认配置为0（Normal）
    */
    int callIndex;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorWarpped {};
    VTensor tTensorCropped {};
    VTensor tTensorResized {};
    VTensor tTensorRotMat {};
};

} // namespace vision
