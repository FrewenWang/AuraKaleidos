#include "FaceQualityDetector.h"

#include <climits>

#include "config/static_config/ref_landmark/ref_face_quality.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"

namespace aura::vision {

FaceQualityDetector::FaceQualityDetector() :
        // 初始化顺序和声明顺序一致
        refFaceWidth(0), refFaceHeight(0),
        inputWidth(0), inputHeight(0),
        rectBox(10, 10, 177, 177),
        qualityIndex(0) {
    TAG = "FaceQualityDetector";
    mPerfTag += TAG;
}

int FaceQualityDetector::init(RtConfig* cfg) {
	mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_QUALITY);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face quality predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        auto shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    init_params();

    V_RET(Error::OK);
}

int FaceQualityDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    // init params
    V_CHECK_COND_ERR((inputHeight == 0 || inputHeight == 0), Error::MODEL_INIT_ERR, "FaceQuality Not Load Model!");
    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_quality_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_quality_convert_color_after");

    // warpAffine 仿射变换
    if ((*infos)->tTensorWarped.empty()) {
        VTensor warpRotMat;
        const VPoint *landmarks = (*infos)->landmark2D106;
        ImageUtil::get_warp_params(landmarks, get_quality_ref_lmk(), warpRotMat);
        va_cv::warp_affine(request->gray, (*infos)->tTensorWarped, warpRotMat,
                           va_cv::VSize(refFaceWidth, refFaceHeight));
        V_CHECK_COND_ERR((*infos)->tTensorWarped.w == 0 || (*infos)->tTensorWarped.h == 0, Error::PREPARE_ERR,
                         "FaceQuality warpAffine result error!");
        DBG_PRINT_ARRAY((float *) warpRotMat.data, 6, "face_quality_warpRotMat");
    }
    DBG_PRINT_ARRAY((char *) (*infos)->tTensorWarped.data, 50, "face_quality_warp_affine_after");

    // crop
    if ((*infos)->tTensorCropped.empty()) {
        va_cv::crop((*infos)->tTensorWarped, (*infos)->tTensorCropped, rectBox);
    }
    DBG_PRINT_ARRAY((char *) (*infos)->tTensorCropped.data, 50, "face_quality_crop_after");
    DBG_RAW("face_quality_crop_after", TensorConverter::convert_to<cv::Mat>((*infos)->tTensorCropped));

    // resize
    va_cv::resize_normalize((*infos)->tTensorCropped, tTensorResized, va_cv::VSize(inputWidth, inputHeight),
                            0, 0, va_cv::INTER_AREA);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_quality_resize_normalize_after");
    DBG_RAW("face_quality_resize_normalize_after", TensorConverter::convert_to<cv::Mat>(tTensorResized));

    // put data
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_quality_prepare_after");
    DBG_RAW("face_quality_prepare_after", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    V_RET(Error::OK);
}

int FaceQualityDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face quality predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceQualityDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face quality infer result is empty");
    // infer result
    for (int i = 0; i < static_cast<int>(infer_results.size()); i++) {
        auto &output = infer_results[i];
        auto *outData = (float *) output.data;
        //指数归一化
        MathUtils::softmax(outData, softmaxOutput, OUTPUT_DATA_SIZE);
        //指数从小到大排序，取最大索引号
        qualityIndex = MathUtils::argmax((float *) softmaxOutput, OUTPUT_DATA_SIZE);
        DBG_PRINT_ARRAY((float *) softmaxOutput, OUTPUT_DATA_SIZE, "face_quality_post_result "
                                                                  "Index:" + std::to_string(i));
        detectIndexResult[i] = qualityIndex;
    }
    //6分类模型，删除模型直接输出遮挡场景，删除置信度判断
    auto face = *infos;
    //判断图片噪声，0-Normal，1-噪声大
    face->stateNoiseSingle = detectIndexResult[0] > faceQualityNormal ?
                             FaceQualityStatus::F_QUALITY_NOISE_HIGH : FaceQualityStatus::F_QUALITY_NOISE_NORMAL;
    //判断图片清晰度，0-Normal，1-模糊
    face->stateBlurSingle = detectIndexResult[1] > faceQualityNormal ?
                            FaceQualityStatus::F_QUALITY_BLUR_HIGH : FaceQualityStatus::F_QUALITY_BLUR_NORMAL;
    //判断左眼部遮挡，0-Normal，1-遮挡
    face->leftEyeCoverSingle = detectIndexResult[2] > faceQualityNormal ?
                            FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE : FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE_NORMAL;
    //判断右眼部遮挡，0-Normal，1-遮挡
    face->rightEyeCoverSingle = detectIndexResult[3] > faceQualityNormal ?
                            FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE : FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE_NORMAL;
    //判断嘴部遮挡，0-Normal，1-遮挡
    face->stateMouthCoverSingle = detectIndexResult[4] > faceQualityNormal ?
                            FaceQualityStatus::F_QUALITY_COVER_MOUTH_HIGH : FaceQualityStatus::F_QUALITY_COVER_MOUTH_NORMAL;
    //根据Jidu产品需求,删除coverOther
    //判断人脸遮挡，眼部/嘴部/其他部位发生一处遮挡，即判定遮挡；0-Normal，1-遮挡大
    if(face->leftEyeCoverSingle == FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE ||
        face->rightEyeCoverSingle == FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE ||
        face->stateMouthCoverSingle == FaceQualityStatus::F_QUALITY_COVER_MOUTH_HIGH) {
        // leftEyeCover-1, rightEyeCover-1, stateMouthCoverSingle-1
        face->stateFaceCoverSingle = FaceQualityStatus::F_QUALITY_COVER_HIGH;
        VLOGD(TAG, "check out leftEye_cover_status=%d, rightEye_cover_status=%d", face->leftEyeCoverSingle, face->rightEyeCoverSingle);
    } else {
        face->stateFaceCoverSingle = FaceQualityStatus::F_QUALITY_COVER_NORMAL;
    }

    V_RET(Error::OK);
}

void FaceQualityDetector::init_params() {
    VPoint max_point(0, 0);
    VPoint min_point(INT_MAX, INT_MAX);
    /// FIXME @wangbin 进行新模型继承时候，和算法同学确认要求Emotion、Attribute、NoInteractLiving、Quality四个模型使用同样的仿射参照系统一起来。
    auto* ref_lmk = get_quality_ref_lmk();
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        VPoint &point = ref_lmk[i];
        max_point.x = MAX(max_point.x, point.x);
        max_point.y = MAX(max_point.y, point.y);
        min_point.x = MIN(min_point.x, point.x);
        min_point.y = MIN(min_point.y, point.y);
    }

    float face_width = max_point.x - min_point.x;
    float face_height = max_point.y - min_point.y;
    face_width = MAX(face_width, face_height);
    face_height = face_width;

    float dx = face_width * 0.1f - min_point.x;
    float dy = face_height * 0.1f - min_point.y;

    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        ref_lmk[i].x += dx;
        ref_lmk[i].y += dy;
    }

    refFaceHeight = static_cast<int>(face_height * 1.2f);
    refFaceWidth = static_cast<int>(face_width * 1.2f);
}

} // namespace vision