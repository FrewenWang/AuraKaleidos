#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

/**
 * @brief 人脸表情检测功能
 * */
class FaceEmotionDetector : public AbsFaceDetector {
public:
    FaceEmotionDetector();

    virtual ~FaceEmotionDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    /**
     * 初始化检测参数
     */
    void initParams();

    /**
     * 初始化仿射变换参照图片的人脸关键点
     * @param refLandmarks
     */
    void initRefLandmark(VPoint *refLandmarks);

    // 后处理归一化数组的大小
    static const int OUTPUT_DATA_SIZE = 5;

    int refFaceWidth = 0;
    int refFaceHeight = 0;
    /** 模型输出的图像宽高尺寸 */
    int inputWidth = 0;
    int inputHeight = 0;

    /** Rect的左上角与右下角坐标 */
    vision::VRect cropRectBox;

    /** happyThreshold for happy */
    const float happyThreshold = 0.5f;
    /** happyIndex of softmaxEmotionOutput output */
    static const int happyIndex = 1;

    /** 输出的归一化数组，取最大元素下标 */
    float softmaxEmotionOutput[OUTPUT_DATA_SIZE];

    int outputIndex = 0;
    float outputScore = 0.f;

    bool refLandmarkInitialized = false;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorResized{};

};
}//namespace vision
