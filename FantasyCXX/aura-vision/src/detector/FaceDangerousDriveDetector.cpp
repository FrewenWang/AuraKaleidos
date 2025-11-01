#include "FaceDangerousDriveDetector.h"

#include <climits>

#include "config/static_config/ref_landmark/ref_face_dangerous_drive.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"

namespace aura::vision {

FaceDangerousDriveDetector::FaceDangerousDriveDetector()
    : inputWidth(0), inputHeight(0), refFaceWidth(0), refFaceHeight(0),
      /**
       * 危险驾驶分类Index,默认配置为0（Normal）
       */
      // 变量初始化顺序需要和头文件中定义顺序一致
      cropRect(10, 10, 260, 260), dangerIndex(0) {
    TAG = "DangerousDriveDetector";
    mPerfTag += TAG;
}

FaceDangerousDriveDetector::~FaceDangerousDriveDetector() = default;

int FaceDangerousDriveDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;

    init_params();

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_DANGEROUS_DRIVING);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Dangerous driving predictor not registered!");

    const auto &modelInputList = _predictor->get_input_desc();
    if (!modelInputList.empty()) {
        const auto &shape = modelInputList[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    V_RET(Error::OK);
}

int FaceDangerousDriveDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    // init params
    V_CHECK_COND_ERR((inputHeight == 0 || inputHeight == 0), Error::MODEL_INIT_ERR, "FaceDanger Not Load Model!");
    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_danger_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_danger_convert_color_after");

    // warpAffine
    const VPoint *landmarks = (*infos)->landmark2D106;
    ImageUtil::get_warp_params(landmarks, get_dangerous_drive_ref_lmk(), _rotation_matrix);
    V_CHECK_COND(_rotation_matrix.empty(), Error::PREPARE_ERR, "FaceDanger detector get warp affine matrx error!");
    DBG_PRINT_ARRAY((float *) _rotation_matrix.data, 6, "face_danger_prepare_rot_mat");
    va_cv::warp_affine(request->gray, tensorWarped, _rotation_matrix,
                       va_cv::VSize(refFaceWidth, refFaceHeight));
    V_CHECK_COND_ERR(tensorWarped.w == 0 || tensorWarped.h == 0, Error::PREPARE_ERR, "FaceDanger warpAffine result error!");
    DBG_PRINT_ARRAY((char *) tensorWarped.data, 50, "face_danger_warp_affine_after");

    // crop
    va_cv::crop(tensorWarped, tensorCropped, cropRect);
    DBG_PRINT_ARRAY((char *) tensorCropped.data, 50, "face_danger_crop_after");

    // resize and normalize
    va_cv::resize_normalize(tensorCropped, tensorResized, va_cv::VSize(inputWidth, inputHeight),
                            0, 0, va_cv::INTER_AREA);
    DBG_PRINT_ARRAY((float*)tensorResized.data, 50, "face_danger_resize_normalize_after");

    // put data
    prepared.clear();
    prepared.emplace_back(tensorResized);
    DBG_PRINT_ARRAY((float*)tensorResized.data, 50, "face_danger_prepare_after");
    DBG_RAW("face_danger_prepare_after", TensorConverter::convert_to<cv::Mat>(tensorResized));
    V_RET(Error::OK);
}

int FaceDangerousDriveDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Dangerous driving predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceDangerousDriveDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Dangerous driving infer result is empty");
    auto &output = infer_results[0];
    auto out_len = output.size();
    auto *out_data = (float *) output.data;
    //cancel smoke burning state check
    /*auto &smokeOutput = infer_results[1];
    auto *smokeData = (float *) smokeOutput.data;*/
    DBG_PRINT_ARRAY(out_data, out_len, "Dangerous driving infer result");
    /**
     * 指数归一化
     */
    MathUtils::softmax(out_data, softmaxDangerOutput, OUTPUT_DATA_SIZE);
    //cancel smoke burning state check
    //MathUtils::softmax(smokeData, softmaxSmokeOutput, SMOKE_OUTPUT_SIZE);
    /**
     * 指数从小到大排序，取最大索引号
     */
    dangerIndex = MathUtils::argmax((float *) softmaxDangerOutput, out_len);
    //smokeIndex = MathUtils::argmax((float *) softmaxSmokeOutput, SMOKE_OUTPUT_SIZE);
    auto *face = *infos;
    // 记录7分类结果对应的得分，在玩手机检测中作为前置判断条件
    face->dangerDriveConfidence = softmaxDangerOutput[dangerIndex];
    switch (dangerIndex) {
        case F_DANGEROUS_NONE:
            face->stateDangerDriveSingle = F_DANGEROUS_NONE;
            break;
        case F_DANGEROUS_SILENCE:
            face->stateDangerDriveSingle = F_DANGEROUS_SILENCE;
            break;
        case F_DANGEROUS_SMOKE: {
            // 获取抽烟行为的置信度，如果低于阈值，判断正常驾驶
            float smokeScore = softmaxDangerOutput[1];
            //float burningScore = softmaxSmokeOutput[1];
            if (smokeScore > smokeConfidence) {
                face->stateDangerDriveSingle = F_DANGEROUS_SMOKE;
            } else {
                face->stateDangerDriveSingle = F_DANGEROUS_NONE;
            }
            face->stateSmokeBurningSingle = SmokingBurningStatus::F_SMOKE_BURNING_UNKNOWN;
            VLOGD(TAG, "danger_drive[%ld] smoke stateSingle:[%d],score:[%f],threshold:[%f]", face->id,
                  face->stateDangerDriveSingle, smokeScore, smokeConfidence);
            break;
        }
        case F_DANGEROUS_OPEN_MOUTH:
            face->stateDangerDriveSingle = F_DANGEROUS_OPEN_MOUTH;
            break;
        case F_DANGEROUS_MASK_COVER:
            face->stateDangerDriveSingle = F_DANGEROUS_MASK_COVER;
            break;
        case F_DANGEROUS_COVER_MOUTH:
            face->stateDangerDriveSingle = F_DANGEROUS_COVER_MOUTH;
            break;
        case F_DANGEROUS_DRINK:
            face->stateDangerDriveSingle = F_DANGEROUS_DRINK;
            break;
        default:
            face->stateDangerDriveSingle = F_DANGEROUS_NONE;
            break;
    }
    V_RET(Error::OK);
}

void FaceDangerousDriveDetector::init_params() {
    auto excludes = [&](int index) {
        return std::find(_g_excluded_ref_lmk_points.begin(),
                         _g_excluded_ref_lmk_points.end(),
                         index)
               != _g_excluded_ref_lmk_points.end();
    };
    VPoint max_point(0, 0);
    VPoint min_point(INT_MAX, INT_MAX);

    auto *ref_lmk = get_dangerous_drive_ref_lmk();

    // filter the points needed for dms ROI
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        if (excludes(i)) {
            continue;
        }

        auto &point = ref_lmk[i];
        max_point.x = MAX(max_point.x, point.x);
        max_point.y = MAX(max_point.y, point.y);
        min_point.x = MIN(min_point.x, point.x);
        min_point.y = MIN(min_point.y, point.y);
    }

    float face_width = max_point.x - min_point.x;
    float face_height = max_point.y - min_point.y;
    face_width = MAX(face_width, face_height);
    face_height = face_width;

    float dx = face_width * 0.4f - min_point.x;
    float dy = face_height * 0.4f - min_point.y;
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        ref_lmk[i].x += dx;
        ref_lmk[i].y += dy;
    }

    refFaceHeight = static_cast<int>(face_height * 1.8f);
    refFaceWidth = static_cast<int>(face_width * 1.8f);
}

} // namespace vision