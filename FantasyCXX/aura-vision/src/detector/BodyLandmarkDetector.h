#pragma once

#include "AbsBodyDetector.h"
#include "vision/core/bean/BodyInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class BodyLandmarkDetector : public AbsBodyDetector {
public:
    BodyLandmarkDetector();
    ~BodyLandmarkDetector() override;

	int init(RtConfig *cfg) override;

protected:
	int prepare(VisionRequest *request, BodyInfo **infos, TensorArray &prepared) override;
	int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;
	int post(VisionRequest *request, TensorArray &infer_results, BodyInfo **infos) override;

private:
    void predictBodyLocation(VPoint& leftUp, VPoint& rightDown, VPoint& rectCenter, int frameWidth, int frameHeight);

    // 模型输入图像的宽高
    int mInputWidth = 0;
    int mInputHeight = 0;
    // 输入给模型的每个body框的尺寸
    float bodyRectWidth = 0.f;
    float bodyRectHeight = 0.f;
    // predict body rect box w/h ratio
    const float rectWidthRatio = 0.5f;
    const float rectLTHeightRatio = 0.2f;
    const float rectRBHeightRatio = 0.8f;
    const float bodyExtendRatio = 1.5f;
    /**
     * 算法最新置信度阈值：0.1 待测试验证。
     */
    const float bodyLandmarkThreshold = 0.1f;
    // tmp point value
    float minPointX = 0.f;
    float minPointY = 0.f;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped {};
    VTensor tTensorResized {};
//    VTensor tTensorResizedFp32 {};
};

}
