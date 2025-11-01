#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class FaceMouthLandmarkDetector : public AbsFaceDetector {
public:
    FaceMouthLandmarkDetector();

    ~FaceMouthLandmarkDetector() override;

    int init(RtConfig *cfg) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    /**
     * mouth landmark模型推理
     * @param request
     * @param result
     * @param face
     * @return  OK or Error
     */
    int doDetectInner(VisionRequest *request, VisionResult *result, FaceInfo **face);

    // 模型输入图像的宽高
    int mInputWidth;
    int mInputHeight;
    // 输入给模型的每个mouth框的尺寸(原始框缩放为1.2倍的宽高)
    float mouthExtendWidth;
    float mouthExtendHeight;
    // mouth landmark 检测框左上角，右下角和中心点
    VPoint minPoint;
    VPoint maxPoint;
    VPoint mouthPoint;
    VPoint mouthCenter;
    /**
     *face mouth landmark检测框拓展系数
     */
    const float extendBoxRatio = 1.2f;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped{};
    VTensor tTensorResized{};
    // 前后处理定义的全局张量数据
    VTensor tensorPrepare;
};

}
