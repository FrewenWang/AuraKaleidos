#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class FaceLandmarkDetector : public AbsFaceDetector {
public:
    FaceLandmarkDetector();

    ~FaceLandmarkDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void resize_box(int width, int height, FaceInfo *face);

    void match_template(VisionRequest *request, FaceInfo *face_info);

    void get_face_template(VisionRequest *request, FaceInfo *face);

    bool check_roi(FaceInfo *face);

    // 模型输入图像的宽高
    int mInputWidth;
    int mInputHeight;
    // 输入给模型的每个人脸框的尺寸(原始人脸框缩放为1.2倍的宽高)
    float faceRectExtendWidth;
    float faceRectExtendHeight;

    // 判断摄像头存在镜像反转
    bool cameraMirror = true;
    /**
     * 模型输出是够有人脸的置信度阈值
     * 上层业务可能根据业务需要修改
     * 比如：
     * 1.人脸注册或者人脸支付的时候设置置信度比较高
     * 2.人脸识别的时候设置置信度比较低
     */
    float lmkThreshold;
    // 2D landmark output raw degrees for mirror calculate (in degrees)
    VAngle calculateDeflection;
    // 模型输出 - 头部偏转角（pitch / yaw / roll）在输出向量（数组）中的下标索引
    const short lmkPitchIndex = 213;
    const short lmkYawIndex = 214;
    const short lmkRollIndex = 215;
    // 模型输出 - 眼睛开闭置信度
    const short lmkEyeCloseThresholdIndex = 216;

    // 配置开关 - 是否使用模版匹配（该文件内使用）
    const bool faceLandmarkTemplateMatchSwitch = false;
    // 配置开关 - 是否使用模版匹配
    bool _use_template_match;

    const int _k_template_img_scale = 8;
    const float _k_input_and_output_rect_verlap_threshold = 0.3f;
    VTensor _template_matrix;
    // 根据算法建议，置信度设置0.15
    const float _k_small_face_threshold = 0.15f;
    // 后处理参考角度
    const float REF_BASE_HEAD_DEFLECTION = 0.0f;
    const float REF_PITCH_HEAD_DEFLECTION = 5.0f;
    const float REF_YAW_HEAD_DEFLECTION = 7.5f;
    /**
     *face landmark检测框拓展系数
     */
    const float extendBoxRatio = 1.2f;
    /**
     * 小人脸的最小的人脸
     */
    const float smallFaceRectMinimumLength = 95;
    const float smallFaceRectPitchThreshold = 10;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped{};
    VTensor tTensorResized{};
};

}
