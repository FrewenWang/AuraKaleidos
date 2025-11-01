#pragma once

#include "AbsGestureDetector.h"
#include "vision/core/bean/GestureInfo.h"
#include "opencv2/core/core.hpp"

namespace aura::vision{

class GestureLandmarkDetector : public AbsGestureDetector {
public:
    GestureLandmarkDetector();
    ~GestureLandmarkDetector() override;

    int init(RtConfig* cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, GestureInfo** infos, TensorArray& prepared) override;
    int process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) override;
    int post(VisionRequest *request, TensorArray& inferResults, GestureInfo** infos) override;

    int prepare(VisionRequest &request, GestureInfo &gesture, std::vector<cv::Mat> &input);
    int process(VisionRequest &request, std::vector<cv::Mat> &input, std::vector<cv::Mat> &output);
    int post(VisionRequest &request, std::vector<cv::Mat> &output, GestureInfo &gesture);

private:
    static void decide_left_or_right_five(GestureInfo* gesture);

    // 模型输入图像的宽高
    int mInputWidth;
    int mInputHeight;

    VTensor mImgCropped {};
    VTensor mImgResized {};

    // 手势关键点数量
    const int mLandmarkCount = 21;
    //gestureType num
    const int gestureTypeNum = 12;

    /** Landmark 检测框拓展系数 */
    const float mExtendBoxRatio = 1.2f;
};

} // namespace vision
