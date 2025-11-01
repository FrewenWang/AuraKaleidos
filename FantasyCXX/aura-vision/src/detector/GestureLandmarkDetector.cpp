#include "GestureLandmarkDetector.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "vacv/cv.h"

using namespace std;

namespace aura::vision {

GestureLandmarkDetector::GestureLandmarkDetector()
        : mInputWidth(0), mInputHeight(0) {
    TAG = "GestureLandmarkDetector";
    mPerfTag += TAG;
}

int GestureLandmarkDetector::init(RtConfig* cfg) {
	mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_GESTURE_LANDMARK);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Gesture landmark predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (!model_input_list.empty()) {
        auto shape = model_input_list[0].shape();
        mInputWidth = shape.w();
        mInputHeight = shape.h();
    }
    V_RET(Error::OK);
}

GestureLandmarkDetector::~GestureLandmarkDetector() = default;

int GestureLandmarkDetector::doDetect(VisionRequest *request, VisionResult *result) {
    if (mInputHeight == 0 || mInputWidth == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    auto gCount = V_TO_INT(mRtConfig->gestureNeedDetectCount);
    GestureInfo **gInfos = result->getGestureResult()->gestureInfos;
    for (int i = 0; i < gCount; ++i) {
        GestureInfo *gesture = gInfos[i];
        V_CHECK_CONT(gesture->id == 0);
        // 当检测到的手势框为打电话的手势框时，不再进行手势landmark的检测
        V_CHECK_CONT(gesture->rectType != RectType::G_RECT_TYPE_GESTURE);
        TensorArray prepared;
        TensorArray predicted;
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
            V_CHECK_CONT(prepare(request, &gesture, prepared) != 0);
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
            V_CHECK_CONT(process(request, prepared, predicted) != 0);
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
            V_CHECK_CONT(post(request, predicted, &gesture) != 0);
        }
    }
    // sort by type probability
    std::sort(gInfos, gInfos + gCount,
              [](GestureInfo *gest1, GestureInfo *gest2) { return gest1->typeConfidence > gest2->typeConfidence; });
    V_RET(Error::OK);
}

// ---------------------------------------------------------------------------------------------------------------------

int GestureLandmarkDetector::prepare(VisionRequest &request, GestureInfo &gesture, std::vector<cv::Mat> &input) {
    // init params
    //// Step : 将框放大为原来的1.2倍
    MathUtils::resizeExtendBox(gesture.rectLT, gesture.rectRB, gesture.rectCenter,
                               request.width, request.height, mExtendBoxRatio);

    // 逻辑保护：保证框左上坐标和右下坐标落在图像区域内
    gesture.rectLT.x = CLAMP(gesture.rectLT.x, 0, request.width);
    gesture.rectLT.y = CLAMP(gesture.rectLT.y, 0, request.height);
    gesture.rectRB.x = CLAMP(gesture.rectRB.x, 0, request.width);
    gesture.rectRB.y = CLAMP(gesture.rectRB.y, 0, request.height);

    // convert color
    request.convertFrameToGray();

    //// Step : 裁剪框图像
    VRect roiRect(gesture.rectLT.x, gesture.rectLT.y, gesture.rectRB.x, gesture.rectRB.y);

    cv::Mat croppedImg;
    //    va_cv::crop(request.gray, croppedImg, roiRect);
    //    va_cv::crop(request.frameTensor, (int) FrameFormat::YUV_422_UYVY, cropImg, roiRect);

    //// Step : Resize
    cv::Mat resizedImg;
    va_cv::resize_normalize(croppedImg, resizedImg,
                            va_cv::VSize(mInputWidth, mInputHeight),
                            0, 0, va_cv::INTER_CUBIC);

    // emplace data
    input.clear();
    input.emplace_back(resizedImg);
    V_RET(Error::OK);
}

int GestureLandmarkDetector::process(VisionRequest &request, vector<cv::Mat> &input, vector<cv::Mat> &output) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Gesture landmark predictor not registered!");
    V_CHECK(_predictor->predict(input, output, mPerfUtil));
    V_RET(Error::OK);
}

int GestureLandmarkDetector::post(VisionRequest &request, vector<cv::Mat> &output, GestureInfo &gesture) {
    V_CHECK_COND(output.size() < 2, Error::INFER_ERR, "Gesture landmark infer results size error");

    auto matLmk = output[0];  // 手势关键点
    auto matType = output[1]; // 手势类型

    auto *typeData = (float *) matType.data;
    auto *lmkData = (float *) matLmk.data;
//    DBG_PRINT_ARRAY(typeData, matType.size(), "gesture_type");
//    DBG_PRINT_ARRAY(lmkData, matLmk.size(), "gesture_landmark");

    //// 获取手势类型
    float *typeThreshold = std::max_element(typeData, typeData + 11);
    int gestureType = static_cast<int>(std::distance(typeData, typeThreshold));

    if (*typeThreshold > mRtConfig->gestureLandmarkThreshold) {
        gesture.staticTypeSingle = gestureType;
        VLOGD(TAG, "Gesture type=%d", static_cast<int>(gestureType));
    } else {
        gesture.clear_all();
    }
    gesture.typeConfidence = *typeThreshold;
    VLOGD(TAG, "Gesture staticTypeSingle=[%d],typeConf=[%f], confThreshold=[%f]", gesture.staticTypeSingle,
          gesture.typeConfidence, mRtConfig->gestureLandmarkThreshold);

    //// 获取手势关键点
//    float lmkConf = lmkData[0]; // 第一个元素表示：置信度
//    if (lmkConf < mRtConfig->gestureLandmarkThreshold) {
//        gesture.clear_all();
//        VA_RET(Error::GESTURE_LMK_ERR);
//    }
//    gesture.landmarkConfidence = lmkConf;
    VPoint *landmark21 = gesture.landmark21;
    float handWidth = gesture.rectRB.x - gesture.rectLT.x;
    float handHeight = gesture.rectRB.y - gesture.rectLT.y;
    for (int i = 0; i < mLandmarkCount; i++) {
        VPoint &point = landmark21[i];
        point.x = lmkData[2 * i + 0] * handWidth / mInputWidth + gesture.rectLT.x;
        point.y = lmkData[2 * i + 1] * handHeight / mInputHeight + gesture.rectLT.y;
    }

//    decide_left_or_right_five(gesture);
//    DBG_PRINT_GEST_LMK(gesture);

    V_RET(Error::OK);
}

// ---------------------------------------------------------------------------------------------------------------------

int GestureLandmarkDetector::prepare(VisionRequest *request, GestureInfo **infos, TensorArray &prepared) {
    GestureInfo *gInfo = *infos;


    //// Step : 将框放大为原来的1.2倍
    MathUtils::resizeExtendBox(gInfo->rectLT, gInfo->rectRB, gInfo->rectCenter,
                               request->width, request->height, mExtendBoxRatio);

    // 逻辑保护：保证框左上坐标和右下坐标落在图像区域内
    gInfo->rectLT.x = CLAMP(gInfo->rectLT.x, 0, request->width);
    gInfo->rectLT.y = CLAMP(gInfo->rectLT.y, 0, request->height);
    gInfo->rectRB.x = CLAMP(gInfo->rectRB.x, 0, request->width);
    gInfo->rectRB.y = CLAMP(gInfo->rectRB.y, 0, request->height);

    //// Step : 格式转换 UYVY -> GRAY
    request->convertFrameToGray();

    //// Step : 裁剪框图像
    VRect roiRect(gInfo->rectLT.x, gInfo->rectLT.y, gInfo->rectRB.x, gInfo->rectRB.y);
//    VLOGD(TAG, "Gesture crop rect=[%f, %f, %f, %f]", roiRect.left, roiRect.top, roiRect.right, roiRect.bottom);

    va_cv::crop(request->gray, mImgCropped, roiRect);

    //// Step : Resize
    va_cv::resizeNoNormalize(mImgCropped, mImgResized, {mInputWidth, mInputHeight});
    /*va_cv::resize(mImgCropped, mImgResized,
                            va_cv::VSize(mInputWidth, mInputHeight),
                            0, 0, va_cv::INTER_LINEAR);
    VTensor input = mImgResized.changeDType(DType::FP32);*/

    //// Step : Emplace image
    prepared.clear();
    prepared.emplace_back(mImgResized);
    V_RET(Error::OK);

//    cv::Mat crop;
//    crop.create(croppedImg.h, croppedImg.w, CV_8UC1);
//    crop.data = (uchar *)croppedImg.data;
//    cv::imshow("croppedImg", crop);
//    cv::waitKey(0);
//

//    VTensor inputImg = resizedImg.changeLayout(NCHW);
//    DBG_RAW("gesture_lmk_processed", TensorConverter::convert_to<cv::Mat>(processed));
//    DBG_PRINT_ARRAY((float *) processed.data, 100, "gesture_lmk_processed");

//    DBG_READ_RAW("/home/baiduiov/work/vision-space/OriginalModel/GestureLandmark/20221205V22/onnx/input.raw",
//                 resizedImg.data, resizedImg.len());
}

int GestureLandmarkDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Gesture landmark predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int GestureLandmarkDetector::post(VisionRequest *request, TensorArray &inferResults, GestureInfo **infos) {
    V_CHECK_COND(inferResults.size() < 2, Error::INFER_ERR, "Gesture landmark infer results size error");

    auto *gesture = *infos;
    auto matLmk = inferResults[0];  // 手势关键点
    auto matType = inferResults[1]; // 手势类型

    auto *typeData = (float *) matType.data;
    auto *lmkData = (float *) matLmk.data;
    DBG_PRINT_ARRAY(typeData, matType.size(), "gesture_type");
    DBG_PRINT_ARRAY(lmkData, matLmk.size(), "gesture_landmark");

    //// 获取手势类型
    float *typeThreshold = std::max_element(typeData, typeData + gestureTypeNum);
    int gestureType = static_cast<int>(std::distance(typeData, typeThreshold));
    //no use case for typeThreshold
    /*if (*typeThreshold > mRtConfig->gestureLandmarkThreshold) {
        gesture->staticTypeSingle = gestureType;
    } else {
        gesture->clear_all();
    }*/
    gesture->staticTypeSingle = gestureType;
    gesture->typeConfidence = *typeThreshold;
    VLOGD(TAG, "Gesture staticTypeSingle=[%d], typeConf=[%f], confThreshold=[%f]", gesture->staticTypeSingle,
          gesture->typeConfidence, mRtConfig->gestureLandmarkThreshold);

    //// 获取手势关键点
    //    float lmkConf = lmkData[0]; // 第一个元素表示：置信度
    //    if (lmkConf < mRtConfig->gestureLandmarkThreshold) {
    //        gesture->clear_all();
    //        VA_RET(Error::GESTURE_LMK_ERR);
    //    }

    // gesture->landmarkConfidence = lmkConf;
    VPoint *landmark21 = gesture->landmark21;
    float handWidth = gesture->rectRB.x - gesture->rectLT.x;
    float handHeight = gesture->rectRB.y - gesture->rectLT.y;
    for (int i = 0; i < mLandmarkCount; i++) {
        VPoint &point = landmark21[i];
        point.x = lmkData[2 * i + 0] * handWidth / mInputWidth + gesture->rectLT.x;
        point.y = lmkData[2 * i + 1] * handHeight / mInputHeight + gesture->rectLT.y;
    }

    // decide_left_or_right_five(gesture);
    // DBG_PRINT_GEST_LMK(gesture);

    V_RET(Error::OK);
}

// 左五或右五判定策略
void GestureLandmarkDetector::decide_left_or_right_five(GestureInfo *gesture) {
    int gest_type = static_cast<int>(gesture->staticTypeSingle);
    if (gest_type != GESTURE_5_RAW && gest_type != GESTURE_4_RAW) {
        return;
    }
    float hand_contour_width =
        fabs(gesture->landmark21[GLM_0_GESTURE_START].x - gesture->landmark21[GLM_12_MIDDLE_FINGER4].x);
    float hand_contour_length =
        fabs(gesture->landmark21[GLM_0_GESTURE_START].y - gesture->landmark21[GLM_12_MIDDLE_FINGER4].y);

    if (hand_contour_width > hand_contour_length) {
        if (gesture->landmark21[GLM_0_GESTURE_START].x > gesture->landmark21[GLM_12_MIDDLE_FINGER4].x) {
            gesture->staticTypeSingle = GESTURE_RIGHT5_RAW;
        } else {
            gesture->staticTypeSingle = GESTURE_LEFT5_RAW;
        }
    }
}

} // namespace vision
