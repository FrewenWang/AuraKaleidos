#pragma once

#include "AbsCameraCoverDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vacv/cv.h"

namespace aura::vision {

/**
 * Source1源摄像头的遮挡的Detector
 * 目前Source1摄像头的遮挡使用的深度学习模型计算
 */
class Source1CameraCoverDetector : public AbsCameraCoverDetector {
public:
    Source1CameraCoverDetector();

    ~Source1CameraCoverDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    /** 后处理归一化数组的大小 */
    static const int OUTPUT_DATA_SIZE = 2;
    /** 模型输入要求的宽高尺寸 */
    int inputWidth = 0;
    int inputHeight = 0;

    /** 均值方差 */
    VTensor meanData;
    VTensor stddevData;
    /** 输出的归一化数组，取最大元素下标  */
    float softmaxCallOutput[OUTPUT_DATA_SIZE];

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorResized{};
};

} // namespace vision
