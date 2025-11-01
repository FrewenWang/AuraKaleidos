#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"

namespace aura::vision {

class FaceQualityDetector : public AbsFaceDetector {
public:
    FaceQualityDetector();

    ~FaceQualityDetector() override = default;

    int init(RtConfig *cfg) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    /**  后处理归一化数组的大小 */
    static const int OUTPUT_DATA_SIZE = 2;
    /**  模型对应的识别场景数，变为5分类 */
    static const int OUTPUT_CASE_SIZE = 5;
    /**  仿射变换参照系人脸宽度和高度 */
    int refFaceWidth;
    int refFaceHeight;
    /**  算法模型需要的输入图片尺寸  */
    int inputWidth;
    int inputHeight;
    /**
     * 人脸框基于仿射变换之后的人脸裁剪的区域框
     */
    vision::VRect rectBox;
    /** 输出的归一化数组，取最大元素下标 */
    float softmaxOutput[OUTPUT_DATA_SIZE];
    /** 输出的检测index数组 */
    short detectIndexResult[OUTPUT_CASE_SIZE];
    /** 输出的归一化数组，取最大元素下标 */
    float detectConfidence[OUTPUT_CASE_SIZE];
    /** 遮挡场景的Index,默认配置为0（Normal）*/
    int qualityIndex;
    /** 人脸遮挡场景，0-normal，非0值-不正常 */
    const int faceQualityNormal = 0;
    const int eyeCoverIndex = 2;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorResized{};
};

} // namespace vision
