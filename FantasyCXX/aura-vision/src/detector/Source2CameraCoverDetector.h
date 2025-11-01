#pragma once

#include "AbsCameraCoverDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vacv/cv.h"

namespace aura::vision {

/**
 * Source2源摄像头的遮挡的Detector
 * 目前Source2摄像头的遮挡使用的CV算法策略完成
 */
class Source2CameraCoverDetector : public AbsCameraCoverDetector {
public:
    Source2CameraCoverDetector();

    ~Source2CameraCoverDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    /** 判断摄像头遮挡的阈值 */
    float cameraCoverThreshold;
    /** 单个网格的阈值 */
    float thresholdMultiZone;
    /** 进行摄像头遮挡的resize的对应。算法同学提供 */
    const int resizeWidth = 100;
    const int resizeHeight = 100;
    va_cv::VSize size = va_cv::VSize(resizeWidth, resizeHeight);
    cv::Size gaussianKSize = cv::Size(3, 3);
    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorResized{};
    cv::Mat imgSrc;
    cv::Mat_<float> imgGaussianBlur;
    cv::Mat imgDifference;
    cv::Scalar scalar;

    /**
     * * 判断摄像头是否遮挡的CV算法策略。
     * 注意：涉及到CV的操作需要进行加速处理
     * @param src 待处理灰度数据
     * @param w  图像宽度
     * @param h  图像高度
     * @return
     */
    float checkCameraCover(const VTensor &src, int w, int h);

    // OMS RGB遮挡的CV算法策略
    // 网格的数量
    const short gridNum = 15;
    /**
     * 单个网格的阈值
     * OMS摄像头的IR模式是1.0
     * OMS摄像头的RGB模式是1.5
     */
    const float thresholdMultiZoneOmsIR = 1.0f;
    const float thresholdMultiZoneOmsRGB = 1.5f;

    const int startGrid = 4;
    const float centerInds = (gridNum - startGrid) * gridNum;
    const int gridWidth = int(resizeWidth / gridNum);
    const int gridHeight = int(resizeHeight / gridNum);


};

} // namespace vision
