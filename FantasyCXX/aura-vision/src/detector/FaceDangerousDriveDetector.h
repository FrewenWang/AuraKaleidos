#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"

namespace aura::vision {

class FaceDangerousDriveDetector : public AbsFaceDetector {
public:
    FaceDangerousDriveDetector();

    ~FaceDangerousDriveDetector() override;

    int init(RtConfig *cfg) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    /**
     * 后处理归一化数组的大小
     * OUTPUT_DATA_SIZE DMSDanger输出大小,SMOKE_OUTPUT_SIZE Smoke输出大小
     */
    static const int OUTPUT_DATA_SIZE = 7;
    static const int SMOKE_OUTPUT_SIZE = 2;
    /** 模型输入图片宽高尺寸 */
    int inputWidth = 0;
    int inputHeight = 0;

    VTensor _rotation_matrix;
    /** 仿射变换的标准人脸的宽高 */
    int refFaceWidth = 0;
    int refFaceHeight = 0;

    /** 输入图片裁剪的Rect区域 */
    vision::VRect cropRect;


    //前后处理定义的全局张量数据
    VTensor tensorWarped;
    VTensor tensorCropped;
    VTensor tensorResized;
    /**
     *输出的归一化数组，取最大元素下标
     * softmaxDangerOutput输出DMSDanger, softmaxSmokeOutput输出Smoke
     */
    float softmaxDangerOutput[OUTPUT_DATA_SIZE];
    float softmaxSmokeOutput[SMOKE_OUTPUT_SIZE];

    /**
     * dangerIndex 危险驾驶分类Index,默认配置为0（Normal）
     * smokeIndex Smoke分类Index,默认配置为0
     */
    short dangerIndex;
    short smokeIndex;

    /**
     * smokeConfidence抽烟行为检测置信度，最新的置信度0.8
     */
    const float smokeConfidence = 0.8f;
    // const float smokeBurningConf = 0.9f;
};

} // namespace vision
