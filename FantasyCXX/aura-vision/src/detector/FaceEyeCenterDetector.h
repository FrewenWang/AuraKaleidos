#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

/**
 * @brief 人脸表情检测功能
 * */
class FaceEyeCenterDetector : public AbsFaceDetector {
public:
    FaceEyeCenterDetector();

    virtual ~FaceEyeCenterDetector() override;

    int init(RtConfig* cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    /**
     * 眼球中心点模型推理
     * @param request
     * @param result
     * @param face
     * @return  OK or Error
     */
    int doDetectInner(VisionRequest *request, VisionResult *result, FaceInfo **face);

    bool _is_left_eye;

    int _ref_face_width;
    int _ref_face_height;

    int _input_width;
    int _input_height;
    /**
     *eyeCenter模型检测框计算的宽与高，命名风格使用大小写
     */
    float boxWidth;
    float boxHeight;
    VPoint _min_point;
    VPoint _max_point;
    /**
     *eyeCenter预处理检测框中心点坐标，命名风格使用大小写
     */
    VPoint boxCenterPoint;

    /**
     *face landmark检测框拓展系数，命名风格使用大小写
     */
    const float extendBoxRatio = 1.4f;

    /** 判断DMS闭眼的上下眼皮距离阈值 */
    const float eyeCloseDistThresholdDms = 6.5f;
    /** 眼睛检测置信度配置 */
    const float eyeDetectedThresholdDms = 0.5f;
    const float eyeDetectedThresholdOms = 0.5f;
    /** 瞳孔检测置信度配置 */
    const float pupilScoreDms = 0.4f;
    const float pupilScoreOms = 0.4f;
    /**
     * 判断是否使用XPERF_TEST(全图输入或局部框图输入），命名风格使用大小写
     */
    bool useXperfTest = false;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped {};
    VTensor tTensorResized {};

    float rectHeight = 0;
    float eyeRef = 0;
    /**
     * 瞳孔中心点里面单独不可复用的计算DMS摄像头遮挡的逻辑
     * @return
     */
    bool checkDmsCameraCover(const VTensor &src, int w, int h);
    /** 判断是否执行过摄像头遮挡的变量 */
    bool hasInvokedCameraCover = false;
    bool isCameraCovered = false;
    /** 单个网格的阈值 */
    float dmsThresholdMultiZone = 1.5;
    /** 将原图resize成100*100的尺寸 **/
    va_cv::VSize coverDstSize = va_cv::VSize(100, 100);
    /** 高斯滤波卷积核的大小 **/
    cv::Size gaussianKSize = cv::Size(3, 3);
    cv::Mat imgSrcMat;
    cv::Mat_<float> imgGaussianBlur;
    cv::Mat imgDifference;
    cv::Scalar scalar;
    VTensor tTensorCoverResized{};
    const short gridNum = 15; // 分割块的大小
    const int gridWidth = int(100 / gridNum);
    const int gridHeight = int(100 / gridNum);
    /// 像素块遮挡个数
    int center_block_num = 0;
    /// DMS 摄像头的遮挡阈值
    float dmsThreshGridRatio = 0.71;
};
}//namespace vision
