
#pragma once

#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/FaceInfo.h"
#include "AbsDetector.h"

namespace aura::vision {

class ImageBrightnessDetector : public AbsDetector<FaceInfo> {

public:
    ImageBrightnessDetector() = default;

    ~ImageBrightnessDetector() override = default;

    int init(RtConfig *cfg) override;

    /**
     * 检查是否过暗
     */
    static bool checkOverDark(VTensor &gray);

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    static constexpr float OVER_DARK_RATIO_THRESHOLD = 0.75f;

};

}
