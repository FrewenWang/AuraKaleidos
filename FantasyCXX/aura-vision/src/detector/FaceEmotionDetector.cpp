#include "FaceEmotionDetector.h"

#include <vector>
#include <math.h>
#include <numeric>

#include "config/static_config/ref_landmark/ref_face_emotion.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"

using namespace aura::vision;

static VTensor mean(1, 1, 1, NHWC, FP32);
static VTensor stddev(1, 1, 1, NHWC, FP32);

FaceEmotionDetector::FaceEmotionDetector():
      // 变量初始化顺序需和声明顺序一致
      refFaceWidth(0), refFaceHeight(0),
      inputWidth(0), inputHeight(0),
      cropRectBox(10, 10, 177, 177) {
            TAG = "FaceEmotionDetector";
            mPerfTag += TAG;
}

FaceEmotionDetector::~FaceEmotionDetector() = default;

int FaceEmotionDetector::init(RtConfig* cfg) {
    mRtConfig = cfg;
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_EMOTION);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face emotion predictor not registered!");

    const auto& model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        const auto& shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    initParams();

    V_RET(Error::OK);
}

int FaceEmotionDetector::doDetect(VisionRequest *request, VisionResult *result) {
    for (int i = 0; i < V_TO_INT(mRtConfig->faceNeedDetectCount); ++i) {
        auto *face = result->getFaceResult()->faceInfos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        // 算法侧要求(jd-7500)：当检测到人脸遮挡时不再进行后续的表情检测，同时将单帧和多帧结果清除
        if (face->stateFaceCoverSingle == F_QUALITY_COVER_HIGH) {
            face->stateEmotionSingle = F_ATTR_UNKNOWN;
            face->stateEmotion = F_ATTR_UNKNOWN;
            face->scoreEmotionSingle = 0.f;
            VLOGD(TAG, "face_emotion[%ld],stateSingle=[%f],index=[%d],score=[%f] cause of face_cover", face->id,
                  face->stateEmotionSingle, outputIndex, outputScore);
            continue;
        }
        TensorArray prepared;
        TensorArray predicted;
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre")
            V_CHECK_CONT_MSG(prepare(request, &face, prepared) != 0, "detect prepare error");
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro")
            V_CHECK_CONT_MSG(process(request, prepared, predicted) != 0, "detect process error");
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos")
            V_CHECK_CONT_MSG(post(request, predicted, &face) != 0, "detect post error");
        }
    }
    V_RET(Error::OK);
}

int FaceEmotionDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    // init params
    if (inputHeight == 0 || inputWidth == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_emotion_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_emotion_convert_color_after");

    // warpAffine 仿射变换
    if ((*infos)->tTensorWarped.empty()) {
        VTensor warpRotMat;
        const VPoint *landmarks = (*infos)->landmark2D106;
        ImageUtil::get_warp_params(landmarks, get_emotion_ref_lmk(), warpRotMat);
        va_cv::warp_affine(request->gray, (*infos)->tTensorWarped, warpRotMat,
                           va_cv::VSize(refFaceWidth, refFaceHeight));
        V_CHECK_COND_ERR((*infos)->tTensorWarped.w == 0 || (*infos)->tTensorWarped.h == 0, Error::PREPARE_ERR,
                         "FaceEmotion warpAffine result error!");
    }

    DBG_PRINT_ARRAY((char *) (*infos)->tTensorWarped.data, 50, "face_emotion_warp_affine_after");

    // crop
    if ((*infos)->tTensorCropped.empty()) {
        va_cv::crop((*infos)->tTensorWarped, (*infos)->tTensorCropped, cropRectBox);
    }
    DBG_PRINT_ARRAY((char*)(*infos)->tTensorCropped.data, 50, "face_emotion_crop_after");
    DBG_RAW("emotion_cropped", TensorConverter::convert_to<cv::Mat>((*infos)->tTensorCropped));

    // resize and normalize
    va_cv::resize_normalize((*infos)->tTensorCropped, tTensorResized, va_cv::VSize(inputWidth, inputHeight),
                            0, 0, va_cv::INTER_AREA, mean, stddev);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 20, "face_emotion_resize_normalize_after");
    DBG_RAW("face_emotion_resize_normalize_after", TensorConverter::convert_to<cv::Mat>(tTensorResized));

    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceEmotionDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs){
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face emotion predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceEmotionDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos){
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face emotion infer result is empty");
    auto &output = infer_results[0];
    auto out_len = output.size();
    auto *out_data = (float *) output.data;

    auto *face = *infos;
    MathUtils::softmax(out_data, softmaxEmotionOutput, OUTPUT_DATA_SIZE);
    DBG_PRINT_ARRAY(softmaxEmotionOutput, OUTPUT_DATA_SIZE, "face_emotion_result");

    outputIndex = MathUtils::argmax(softmaxEmotionOutput, OUTPUT_DATA_SIZE);
    outputScore = softmaxEmotionOutput[outputIndex];

    // 如果分类结果是happy. 则需要判断其置信度是否大约阈值0.5。如果小于，则设置分类结果为other
    // sad/anger/surprise此三个分类不需要判断置信度
    if (((outputIndex == FaceAttributeStatus::F_ATTR_EMOTION_HAPPY) && (outputScore < happyThreshold))) {
        VLOGD(TAG, "face_emotion[%ld],stateSingle=[%f],index=[%d],score=[%f]",
              face->id, face->stateEmotionSingle, outputIndex, outputScore);
        face->stateEmotionSingle = FaceAttributeStatus::F_ATTR_EMOTION_OTHER;
        face->scoreEmotionSingle = outputScore;
    } else {
        face->stateEmotionSingle = outputIndex;
        face->scoreEmotionSingle = outputScore;
    }

    V_RET(Error::OK);
}

void FaceEmotionDetector::initParams() {
    auto *emotion_base_landmarks = get_emotion_ref_lmk();
    initRefLandmark(emotion_base_landmarks);

    //fix mean and stddev in prepare method
    ((float *) mean.data)[0] = 127.5f;
    ((float *) stddev.data)[0] = 128.0f;
}

void FaceEmotionDetector::initRefLandmark(VPoint *refLandmarks) {
    if (refLandmarkInitialized) {
        return;
    }
    /// FIXME @wangbin 进行新模型继承时候，和算法同学确认要求Emotion、Attribute、NoInteractLiving、Quality四个模型使用同样的仿射参照系统一起来。
    float face_hight = refLandmarks[16].y - ImageUtil::get_landmarks_y_min(refLandmarks, {34, 35, 36, 43, 44, 45});
    float face_width = ImageUtil::get_landmarks_x_max(refLandmarks, {32, 31, 30}) -
                       ImageUtil::get_landmarks_x_min(refLandmarks, {0, 1, 2});
    VPoint start(ImageUtil::get_landmarks_x_min(refLandmarks, {0, 1, 2}),
                 ImageUtil::get_landmarks_y_min(refLandmarks, {34, 35, 36, 43, 44, 45}));

    face_width = MAX(face_width, face_hight);
    face_hight = face_width;
    float dx = face_width * 0.1f - start.x;
    float dy = face_hight * 0.1f - start.y;

    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        refLandmarks[i].x += dx;
        refLandmarks[i].y += dy;
    }

    refFaceHeight = face_hight * 1.2f;
    refFaceWidth = face_width * 1.2f;

    refLandmarkInitialized = true;
}

