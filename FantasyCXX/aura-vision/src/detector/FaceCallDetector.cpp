#include "FaceCallDetector.h"

#include <cmath>

#include "config/static_config/ref_landmark/ref_face_phone_call.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"
namespace aura::vision {

FaceCallDetector::FaceCallDetector()
        : defaultDetectLeft(true),
          faceCallRefLmkLeft(new VPoint[LM_2D_106_COUNT]),
          faceCallRefLmkRight(new VPoint[LM_2D_106_COUNT]),
          cropRect(7, 10, 193, 149.5), callIndex(0) {
    TAG = "FaceCallDetector";
    mPerfTag += TAG;
}

int FaceCallDetector::init(RtConfig* cfg) {
	mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_CALL);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Phone call predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        auto shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    init_params();
    V_RET(Error::OK);
}

FaceCallDetector::~FaceCallDetector() {
    delete[] faceCallRefLmkLeft;
    delete[] faceCallRefLmkRight;
}

int FaceCallDetector::doDetect(VisionRequest *request, VisionResult *result) {
    FaceInfo **infos = result->getFaceResult()->faceInfos;
    for (int i = 0; i < V_TO_INT(mRtConfig->faceNeedDetectCount); ++i) {
        auto *face = infos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        // 如果发版模式为产线，且使用左右耳互斥逻辑，则进行左右耳交替检测
        if (mRtConfig->releaseMode == PRODUCT && V_TO_INT(mRtConfig->callUseMutexMode)) {
            // 如果上一帧检测到了结果，则继续检测上一帧的方向，否则使用默认值 _def_detect_left。
            detectLeftEar = defaultDetectLeft;
            if (face->stateCallLeftSingle) {
                detectLeftEar = true;
            } else if (face->stateCallRightSingle) {
                detectLeftEar = false;
            }
            // 当判断完成检测左或右耳朵之后，重置 FaceInfo 中 stateCallLeftSingle and stateCallRightSingle 的值。
            face->stateCallLeftSingle = FaceCallStatus::F_CALL_NONE;
            face->stateCallRightSingle = FaceCallStatus::F_CALL_NONE;
            // 左右耳互斥检测策略，一旦检测到到某侧的耳朵打电话，则会一直检测该侧耳朵
            doDetectInner(request, result, &face);
        } else {
            // 左右耳同时检测
            detectLeftEar = true;
            doDetectInner(request, result, &face);
            detectLeftEar = false;
            doDetectInner(request, result, &face);
        }
    }
    // 每帧检测后结果取反
    defaultDetectLeft = !defaultDetectLeft;
    V_RET(Error::OK);
}

int FaceCallDetector::doDetectInner(VisionRequest *request, VisionResult *result, FaceInfo **face) {
    TensorArray prepared;
    TensorArray predicted;
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre")
        V_CHECK_CONT_MSG(prepare(request, face, prepared) != 0, "face_call prepare error");
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro")
        V_CHECK_CONT_MSG(process(request, prepared, predicted) != 0, "face_call process error");
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos")
        V_CHECK_CONT_MSG(post(request, predicted, face) != 0, "face_call post error");
    }
    V_RET(Error::OK);
}

int FaceCallDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    V_CHECK_COND_ERR((inputHeight == 0 || inputWidth == 0), Error::MODEL_INIT_ERR, "Not load the model!!!");
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_call_prepare_before");

    // convert color
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_call_convert_color_after");
    // warp affine
    const VPoint* landmarks = (*infos)->landmark2D106;
    auto* ref_lmk = detectLeftEar ? faceCallRefLmkLeft : faceCallRefLmkRight;
    ImageUtil::get_warp_params(landmarks, ref_lmk, tTensorRotMat);
    va_cv::warp_affine(request->gray, tTensorWarpped, tTensorRotMat,
                       va_cv::VSize(cropWidth, cropHeight));
    V_CHECK_COND_ERR(tTensorWarpped.empty(), Error::UNKNOWN_FAILURE, "face_call detector prepare error!");
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_call_warp_affine_after");

    // crop after align
    va_cv::crop(tTensorWarpped, tTensorCropped, cropRect);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_call_crop_after");

    // resize and normalize
    static VTensor mean(1, 1, 1, NHWC, FP32);
    static VTensor stddev(1, 1, 1, NHWC, FP32);
    ((float*)mean.data)[0]   = 127.5f;
    ((float*)stddev.data)[0] = 128.0f;
    va_cv::resize_normalize(tTensorCropped, tTensorResized, va_cv::VSize(inputWidth, inputHeight),
                            0, 0, va_cv::INTER_AREA, mean, stddev);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_call_prepare_after");
    DBG_RAW("phone_call", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceCallDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "face_call predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceCallDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "face_call infer result is empty");
    auto &output = infer_results[0];
    auto out_len = output.size();
    auto *out_data = (float *) output.data;
    // 指数归一化
    MathUtils::softmax(out_data, softmaxCallOutput, OUTPUT_DATA_SIZE);
    DBG_PRINT_ARRAY((float *) softmaxCallOutput, out_len, "face_call_post_result");

    auto *face = infos[0];
    // 指数从小到大排序，取最大索引号
    callIndex = MathUtils::argmax((float *) softmaxCallOutput, out_len);
    // model outputs 3 cases: 0：normal, 1：calling, 2：hand_nearby
    scoreFaceCall= softmaxCallOutput[1];
    scoreGestureNearby = softmaxCallOutput[2];
    //set call & gestureNearby strategy if yaw out of angle, useStrategy set false by default
    if(useStrategy && ((!detectLeftEar && face->headDeflection.yaw < rightYawUpper) || (detectLeftEar && face->headDeflection.yaw > leftYawUpper))) {
        VLOGD(TAG, "DMSCall detect out of range,gestureNearbyScore=%f,detect side=%s", scoreGestureNearby, detectLeftEar ? "left" : "right");
        if (callIndex != F_CALL_NONE && scoreFaceCall > callScoreLargeAngle && scoreGestureNearby < callScoreNormalAngle ) {
            callIndex = F_CALL_CALLING;
        } else {
            callIndex = F_CALL_NONE;
        }
    } else {
        if (callIndex != F_CALL_NONE && scoreFaceCall > callScoreNormalAngle) {
            callIndex = F_CALL_CALLING;
        } else {
            callIndex = F_CALL_NONE;
        }
    }
    if (detectLeftEar) {
        face->stateCallLeftSingle = callIndex;
        face->scoreCallLeftSingle = scoreFaceCall;
    } else {
        face->stateCallRightSingle = callIndex;
        face->scoreCallRightSingle = scoreFaceCall;
    }
    V_RET(Error::OK);
}

void FaceCallDetector::init_params() {
    // compute eye width
    VPoint left_eye;
    VPoint right_eye;
    auto *ref_lmk = get_call_ref_lmk();
    left_eye += ref_lmk[FLM_52_L_EYE_TOP_LEFT_QUARTER];
    left_eye += ref_lmk[FLM_53_L_EYE_TOP_RIGHT_QUARTER];
    left_eye += ref_lmk[FLM_56_L_EYE_LOWER_RIGHT_QUARTER];
    left_eye += ref_lmk[FLM_55_L_EYE_LOWER_LEFT_QUARTER];
    left_eye /= 4.f;

    right_eye += ref_lmk[FLM_62_R_EYE_TOP_LEFT_QUARTER];
    right_eye += ref_lmk[FLM_63_R_EYE_TOP_RIGHT_QUARTER];
    right_eye += ref_lmk[FLM_66_R_EYE_LOWER_RIGHT_QUARTER];
    right_eye += ref_lmk[FLM_65_R_EYE_LOWER_LEFT_QUARTER];
    right_eye /= 4.f;

    // compute ear width
    auto eye_width = std::sqrt(std::pow(right_eye.x - left_eye.x, 2) +
                               std::pow(right_eye.y - left_eye.y, 2));
    cropWidth = static_cast<int>(eye_width * 3.f / 2.f + cropBoxBias);
    cropHeight = static_cast<int>(cropWidth * 3.f / 2.f + cropBoxBias);

    // nose position lmk
    refLmkLeftStart.x = static_cast<int>(ref_lmk[FLM_75_NOSE_LEFT_CT1].x - (cropWidth * cropWidthRatio + cropBoxBias));
    refLmkLeftStart.y = static_cast<int>(ref_lmk[FLM_75_NOSE_LEFT_CT1].y - cropHeight * cropBiasRatio + cropBoxBias);

    refLmkRightStart.x = static_cast<int>(ref_lmk[FLM_76_NOSE_RIGHT_CT1].x + cropBoxBias);
    refLmkRightStart.y = static_cast<int>(ref_lmk[FLM_76_NOSE_RIGHT_CT1].y - cropHeight * cropBiasRatio + cropBoxBias);

    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        faceCallRefLmkLeft[i] = ref_lmk[i] - refLmkLeftStart;
        faceCallRefLmkRight[i] = ref_lmk[i] - refLmkRightStart;
    }

    // new size
    cropWidth = cropWidth * cropWidthRatio + 3.f * cropBoxBias;
    cropHeight = cropHeight * cropHeightRatio + cropBoxBias;
}

} // namespace vision
