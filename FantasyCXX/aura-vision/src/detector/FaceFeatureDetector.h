#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "util/DebugUtil.h"

namespace aura::vision {

class FaceFeatureDetector : public AbsFaceDetector {
public:
    FaceFeatureDetector();
    ~FaceFeatureDetector() override;

    int init(RtConfig* cfg) override;

protected:
    void init_params();
//    int prepare(VFrameInfo& frame, FaceInfo** infos, TensorArray& prepared) override;
    int prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) override;
    int process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) override;
    int post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) override;

private:
    int inputWidth;
    int inputHeight;
    VTensor meanData;
    VTensor stddevData;
    // 调试模式下。保存上一帧的特征值进行特征比对的逻辑
#ifdef DEBUG_SUPPORT
    float preFeature[FEATURE_COUNT];
#endif
    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorWarped {};
    VTensor tTensorResized {};
};

} // namespace vision
