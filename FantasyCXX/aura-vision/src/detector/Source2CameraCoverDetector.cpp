#include "Source2CameraCoverDetector.h"

#include "inference/InferenceRegistry.h"
#include "vacv/resize.h"
#include "util/math_utils.h"
#include "opencv2/opencv.hpp"
#include "util/TensorConverter.h"

namespace aura::vision {

Source2CameraCoverDetector::Source2CameraCoverDetector() {
    TAG = "Source1CameraCoverDetector";
    mPerfTag += TAG;
}

int Source2CameraCoverDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    V_RET(Error::OK);
}

Source2CameraCoverDetector::~Source2CameraCoverDetector() {}

int Source2CameraCoverDetector::doDetect(VisionRequest *request, VisionResult *result) {
    // 集度oms摄像头IR/RGB模式从ebd数据中解析，不支持ebd数据的摄像头使用策略判断
    if (!request->isSupportEbd) {
        request->checkIsIrOrRgb();
    }
    // 进行灰度图的转换
    request->convertFrameToGray();
    // 暂无专门的数据结构存储摄像头遮挡的状态，故暂时放在第一个人脸数据里面
    auto face = result->getFaceResult()->faceInfos[0];
    // 判断单帧单帧
    face->scoreCameraCoverSingle = checkCameraCover(request->gray, request->width, request->height);
    if (mRtConfig->cameraLightType == CAMERA_LIGHT_TYPE_IR) {
        cameraCoverThreshold = mRtConfig->cameraCoverThresholdIr;
        thresholdMultiZone = thresholdMultiZoneOmsIR;
    } else {
        cameraCoverThreshold = mRtConfig->cameraCoverThresholdRgb;
        thresholdMultiZone = thresholdMultiZoneOmsRGB;
    }
    // OMS IR和RGB场景都沿用遮挡比例的判定策略
    int center_block_num = 0;
    // block score for multi zones, start from 4(workaround)
    for (int i = startGrid; i < gridNum; ++i) {
        float interval_h_start = i * gridHeight;
        float interval_h_end = (i + 1) * gridHeight;
        for (int j = 0; j < gridNum; ++j) {
            float interval_w_start = j * gridWidth;
            float interval_w_end = (j + 1) * gridWidth;
            // 取一个方形图像
            cv::Mat delta_block = imgDifference(cv::Range(interval_h_start, interval_h_end),
                                                cv::Range(interval_w_start, interval_w_end));
            double mean_delta_block = cv::mean(delta_block)[0];
            if (mean_delta_block < thresholdMultiZone) {
                center_block_num++;
            }
        }
    }
    // ratio of blocked zones
    face->scoreCameraCoverSingle = static_cast<float>(center_block_num / centerInds);
    face->stateCameraCoverSingle = face->scoreCameraCoverSingle >= cameraCoverThreshold ? ImageCoverStatus::F_IMAGE_COVER_BAD : ImageCoverStatus::F_IMAGE_COVER_GOOD;

    V_RET(Error::OK);
}

int Source2CameraCoverDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_RET(Error::OK);
}

int Source2CameraCoverDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_RET(Error::OK);
}

int Source2CameraCoverDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_RET(Error::OK);
}

float Source2CameraCoverDetector::checkCameraCover(const VTensor &src, int w, int h) {
    if (nullptr == src.data || w <= 0 || h <= 0) {
        return 0.0f; // 由以前false修改为0.0f
    }
    tTensorResized.release();
    imgSrc.release();
    imgGaussianBlur.release();
    imgDifference.release();

    va_cv::Resize::resize(src, tTensorResized, size, 0, 0, cv::INTER_AREA);
    if (tTensorResized.empty()) {
        VLOGE(TAG, "resizeGray is empty");
        return 0.0f;
    }
    // 将据格式如果不是float32可以转化成float32
    tTensorResized = tTensorResized.changeDType(FP32);
    // 将resize之后数据转化成Mat
    imgSrc = TensorConverter::convert_to<cv::Mat>(tTensorResized, true);
    // 使用3x3的高斯核高斯模糊滤波器对图像进行平滑处理。
    cv::GaussianBlur(imgSrc, imgGaussianBlur, gaussianKSize, 1000.0);
    // 判断图像差异
    imgDifference = cv::abs(imgSrc - imgGaussianBlur);
    scalar = cv::mean(imgDifference);
    return scalar[0];
}

} // namespace vision
