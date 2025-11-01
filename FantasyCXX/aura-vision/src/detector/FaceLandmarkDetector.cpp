#include "FaceLandmarkDetector.h"

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "vacv/cv.h"
#include "util/TensorConverter.h"

namespace aura::vision {

FaceLandmarkDetector::FaceLandmarkDetector()
        : mInputWidth(0),
          mInputHeight(0),
          faceRectExtendWidth(0.f),
          faceRectExtendHeight(0.f),
          _use_template_match(false) {
    TAG = "FaceLandmarkDetector";
    mPerfTag += TAG;
}

FaceLandmarkDetector::~FaceLandmarkDetector() = default;

int FaceLandmarkDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_LANDMARK);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face landmark predictor not registered!");
    // 获取模型反序列化信息
    const auto &model_input_list = _predictor->get_input_desc();
    if (!model_input_list.empty()) {
        const auto &shape = model_input_list[0].shape();
        mInputWidth = shape.w();
        mInputHeight = shape.h();
    }
    // 设置landmark检测的置信度阈值
    lmkThreshold = mRtConfig->landmarkThresholdNormal;
    // 初始化Camera镜像反转的标志变量
    cameraMirror = V_F_TO_BOOL(mRtConfig->cameraImageMirror);
    V_RET(Error::OK);
}

int FaceLandmarkDetector::doDetect(VisionRequest *request, VisionResult *result) {
    FaceInfo **infos = result->getFaceResult()->faceInfos;
    int faceCount = static_cast<int>(mRtConfig->faceNeedDetectCount);
    for (int i = 0; i < faceCount; ++i) {
        auto *face = infos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        V_CHECK_CONT(!check_roi(face)); // 是否在 ROI 区域外
        TensorArray prepared;
        TensorArray predicted;
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre")
            V_CHECK_CONT_MSG(prepare(request, &face, prepared) != 0, "face_landmark prepare error!");
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro")
            V_CHECK_CONT_MSG(process(request, prepared, predicted) != 0, "face_landmark process error!");
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos")
            V_CHECK_CONT_MSG(post(request, predicted, &face) != 0, "face_landmark post error!");
        }
    }
    V_RET(Error::OK);
}

int FaceLandmarkDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_CHECK_COND_ERR((mInputHeight == 0 || mInputWidth == 0), Error::MODEL_INIT_ERR, "FaceLandmark Not Load Model!");

    auto *face = *infos;
    // 是否开启模板匹配 扩充人脸框  开启模板匹配可以更好的检测到人脸，减少关键点的抖动
    // 通过get_template_matrix获取到模板后,以后都用那张图片进行模板匹配
    if (faceLandmarkTemplateMatchSwitch && _use_template_match) {
        match_template(request, face);
    } else {
        // 逻辑保护：保证人脸框左上坐标和右下坐标落在图像区域内
        resize_box(request->width, request->height, face);
    }
    // 将人脸框缩放为原来的1.2倍
    MathUtils::resizeExtendBox(face->rectLT, face->rectRB, face->rectCenter,
                               request->width, request->height, extendBoxRatio);
    // 逻辑保护：保证人脸框左上坐标和右下坐标落在图像区域内
    face->rectLT.x = CLAMP(face->rectLT.x, 0, request->width);
    face->rectLT.y = CLAMP(face->rectLT.y, 0, request->height);
    face->rectRB.x = CLAMP(face->rectRB.x, 0, request->width);
    face->rectRB.y = CLAMP(face->rectRB.y, 0, request->height);
    // 裁剪人脸区域
    faceRectExtendWidth = face->rectRB.x - face->rectLT.x;
    faceRectExtendHeight = face->rectRB.y - face->rectLT.y;
    V_CHECK_COND(!(faceRectExtendWidth > 0 && faceRectExtendHeight > 0), Error::PREPARE_ERR, "crop face size invalid!");

    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_landmark_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_landmark_cvt_color_after");

    // crop crop算法原始模型使用使用int类型进行裁剪的。所以此处进行转化到int类型
    VRect rect((int) face->rectLT.x, (int) face->rectLT.y, (int) face->rectRB.x, (int) face->rectRB.y);
    if (rect.left < 0 || rect.top < 0 || rect.right > request->gray.w || rect.bottom > request->gray.h) {
        VLOGE(TAG, "init crop rect error of [%f,%f,%f,%f]", rect.left, rect.top, rect.right, rect.bottom);
        V_RET(Error::PREPARE_ERR);
    }

    va_cv::crop(request->gray, tTensorCropped, rect);
    DBG_PRINT_RECT(rect, "face_landmark_crop_rect");
    DBG_PRINT_ARRAY((char *) tTensorCropped.data, 50, "face_landmark_crop_after");

    // resize and no normalize
    va_cv::resizeNoNormalize(tTensorCropped, tTensorResized, {mInputWidth, mInputHeight});

    // 调试逻辑：存储人脸框前处理数据、打印前处理数据、存储前处理图片
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_landmark_prepare_after");
    DBG_IMG("face_landmark_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    // 调试逻辑：读取前处理之后的RAW数据
    // DBG_READ_RAW("./debug_save/face_landmark_prepare.bin", tTensorResized.data, tTensorResized.len());

    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceLandmarkDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face landmark predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceLandmarkDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face landmark infer result is empty");
    auto &output = infer_results[0];
    auto *out_data = (float *) output.data;
    auto *face = *infos;

    if (V_TO_INT(mRtConfig->landmarkDetectScene) == LMK_SCENARIO_NORMAL) {
        lmkThreshold = mRtConfig->landmarkThresholdNormal;
    } else {
        lmkThreshold = mRtConfig->landmarkThresholdFaceid;
    }

    face->landmarkConfidence = out_data[0]; // 置信度
    // 判断置信度的阈值
    if (face->landmarkConfidence < lmkThreshold) {
        if (faceLandmarkTemplateMatchSwitch) {
            _use_template_match = false;
            _template_matrix.release();
        }
        // 检测到的人脸如果关键点检测置信度过低，标记人脸无效，状态设置为UNKNOWN
        face->faceType = F_TYPE_UNKNOWN;
        VLOGW(TAG, "face_landmark[%s] conf too low! conf=%f,threshold=%f",
              std::to_string(face->id).c_str(), face->landmarkConfidence,
              lmkThreshold);
        V_RET(Error::FACE_LMK_ERR);
    }

    auto min_x = static_cast<float>(request->width);
    auto min_y = static_cast<float>(request->height);
    float max_x = 0.f;
    float max_y = 0.f;
    for (int j = 0; j < LM_2D_106_COUNT; j++) {
        VPoint &p = face->landmark2D106[j];
        p.x = out_data[2 * j + 1] * faceRectExtendWidth / mInputWidth + face->rectLT.x;
        p.y = out_data[2 * j + 2] * faceRectExtendHeight / mInputHeight + face->rectLT.y;
        if (min_x > p.x) min_x = p.x;
        if (min_y > p.y) min_y = p.y;
        if (max_x < p.x) max_x = p.x;
        if (max_y < p.y) max_y = p.y;
    }
    DBG_PRINT_FACE_LMK(face);
    // 兼容人脸跟随策略，如果人脸框的值为零
    // 使用landmark的来重新定义人脸框的置信度和坐标
    if (V_F_EQUAL_ZERO(face->rectConfidence)) {
        face->rectConfidence = face->landmarkConfidence;
    }
    face->rectLT.x = CLAMP(min_x, 0, request->width);
    face->rectLT.y = CLAMP(min_y, 0, request->height);
    face->rectRB.x = CLAMP(max_x, 0, request->width);
    face->rectRB.y = CLAMP(max_y, 0, request->height);
    // clear yaw and pitch of calculateDeflection cache
    calculateDeflection.clear();
    // get raw output of yaw and pitch
    calculateDeflection.yaw = out_data[lmkYawIndex] * RADIAN_2_ANGLE_FACTOR;
    calculateDeflection.pitch = out_data[lmkPitchIndex] * RADIAN_2_ANGLE_FACTOR;
    face->eyeCloseConfidence = out_data[lmkEyeCloseThresholdIndex]; // 睁闭眼概率

    // instead of output directly by model, now use left/right eye center to calculate roll
    float leftEyeX = 0;
    float leftEyeY = 0;
    float rightEyeX = 0;
    float rightEyeY = 0;
    for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_70_R_EYE_CENTER; i++) {
        rightEyeX += face->landmark2D106[i].x;
        rightEyeY += face->landmark2D106[i].y;
    }
    for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_60_L_EYE_CENTER; i++) {
        leftEyeX += face->landmark2D106[i].x;
        leftEyeY += face->landmark2D106[i].y;
    }
    leftEyeX /= 9.0f;
    leftEyeY /= 9.0f;
    rightEyeX /= 9.0f;
    rightEyeY /= 9.0f;
    // in case of dx equals to 0
    float dx = rightEyeX - leftEyeX + 1e-6;
    float dy = rightEyeY - leftEyeY;
    face->optimizedHeadDeflection.roll = static_cast<float>(std::atan2(dy, dx) * ANGEL_180 / M_PI);
    face->headDeflection.roll = face->optimizedHeadDeflection.roll;
    if (calculateDeflection.pitch > REF_BASE_HEAD_DEFLECTION) {
        face->optimizedHeadDeflection.pitch = calculateDeflection.pitch + REF_PITCH_HEAD_DEFLECTION;
    } else {
        face->optimizedHeadDeflection.pitch = calculateDeflection.pitch;
    }
    if (calculateDeflection.yaw > REF_BASE_HEAD_DEFLECTION) {
        face->optimizedHeadDeflection.yaw = calculateDeflection.yaw + REF_YAW_HEAD_DEFLECTION;
    } else {
        face->optimizedHeadDeflection.yaw = calculateDeflection.yaw - REF_YAW_HEAD_DEFLECTION;
    }

    // 算法模型原始输出是：左正右负、下正上负
    // 需要转化为标准的坐标系原则：即遵循用户体验的：左正右负、上正下负
    // 所以pitch角度要取反，yaw和roll角度如果是镜像的(用户体验方向和图像方向一致)则不需要取反，否则则也需要进行取反
    if (cameraMirror) {
        face->headDeflection.yaw =  calculateDeflection.yaw;
    } else {
        // 如果是镜像反转的摄像头，则需要跟pitch一样进行取反
        face->headDeflection.yaw = -calculateDeflection.yaw;
        face->optimizedHeadDeflection.yaw = -(face->optimizedHeadDeflection.yaw);
        // 如果是镜像反转的摄像头，roll也进行取反
        face->headDeflection.roll = -(face->headDeflection.roll);
        face->optimizedHeadDeflection.roll = -(face->optimizedHeadDeflection.roll);
    }
    face->headDeflection.pitch = -calculateDeflection.pitch;
    face->optimizedHeadDeflection.pitch = -(face->optimizedHeadDeflection.pitch);
    // for large angle case log print
    VLOGI(TAG,
          "face_landmark[%s],conf[%f],threshold[%f],optimized_rect=[%f,%f,%f,%f],head_deflection=[%f,%f,%f],"
          "optimized_head_deflection=[%f,%f,%f],closeEyeConf=[%f],threshold=[%f],cameraMirror=[%d],"
          "pre5LM[(%f, %f) (%f, %f) (%f, %f) (%f, %f) (%f, %f)]",
          std::to_string(face->id).c_str(), face->landmarkConfidence, lmkThreshold, face->rectLT.x, face->rectLT.y,
          face->rectRB.x, face->rectRB.y, face->headDeflection.yaw, face->headDeflection.pitch,
          face->headDeflection.roll, face->optimizedHeadDeflection.yaw, face->optimizedHeadDeflection.pitch,
          face->optimizedHeadDeflection.roll, face->eyeCloseConfidence, lmkThreshold, cameraMirror,
          face->landmark2D106[59].x, face->landmark2D106[59].y, face->landmark2D106[69].x, face->landmark2D106[69].y,
          face->landmark2D106[74].x, face->landmark2D106[74].y, face->landmark2D106[86].x, face->landmark2D106[86].y,
          face->landmark2D106[90].x, face->landmark2D106[90].y);

    if (faceLandmarkTemplateMatchSwitch) {
        get_face_template(request, face);
    }

    // 计算重叠分值 TODO 此代码的处理逻辑的思想?
    // landmark点阵和人脸检测拓展后的方框面积比
    float landmark_max_w = face->rectRB.x - face->rectLT.x;
    float landmark_max_h = face->rectRB.y - face->rectLT.y;
    float verlap_rate = (landmark_max_h * landmark_max_w) / (faceRectExtendWidth * faceRectExtendHeight);
    // VLOGD(TAG, "face landmark and crop Overlap rate = %f", verlap_rate);
    if (verlap_rate < _k_input_and_output_rect_verlap_threshold &&
        face->landmarkConfidence < _k_small_face_threshold) {
        if (faceLandmarkTemplateMatchSwitch) {
            _use_template_match = false;
            _template_matrix.release();
        }
        // landmark点阵和人脸检测拓展后的方框面积比小于阈值，标记人脸无效，状态设置为UNKNOWN
        face->faceType = F_TYPE_UNKNOWN;
        VLOGW(TAG, "face landmark small && threshold low=%f", lmkThreshold);
        V_RET(Error::FACE_LMK_ERR);
    }
    // 按照算法同学最新策略，理论上人脸框最宽边只要小于110就应该被过滤掉
    // 除非,pitch角度大于10度，可以将人脸最宽边可以放宽到最小95.
    float faceLength = std::max(face->rectRB.x - face->rectLT.x, face->rectRB.y - face->rectLT.y);
    if (faceLength < mRtConfig->faceRectMinPixelThreshold) {
        if (faceLength > smallFaceRectMinimumLength && face->headDeflection.pitch > smallFaceRectPitchThreshold) {
            V_RET(Error::OK);
        } else {
            if (faceLandmarkTemplateMatchSwitch) {
                _use_template_match = false;
                _template_matrix.release();
            }
            // landmark点阵和人脸检测拓展后的方框面积比小于阈值，标记人脸无效，状态设置为UNKNOWN
            face->faceType = F_TYPE_UNKNOWN;
            VLOGW(TAG, "face[%ld] landmark too small!!! faceLength=%f,pitch=%f", face->id, faceLength,
                  face->headDeflection.pitch);
            V_RET(Error::FACE_LMK_ERR);
        }
    }
    V_RET(Error::OK);
}

bool FaceLandmarkDetector::check_roi(FaceInfo *face) {
    if (!V_F_TO_BOOL(mRtConfig->useDriverRoiRectFilter)) {
        return true;
    }

    auto roi_lt_x = mRtConfig->driverRoiLeftTopX;
    auto roi_lt_y = mRtConfig->driverRoiLeftTopY;
    auto roi_rb_x = mRtConfig->driverRoiRightBottomX;
    auto roi_rb_y = mRtConfig->driverRoiRightBottomY;

    if (!(face->rectLT.x >= roi_lt_x && face->rectLT.y >= roi_lt_y &&
          face->rectRB.x <= roi_rb_x && face->rectRB.y <= roi_rb_y)) {
        VLOGE(TAG, "detected face is not within the set area, drop the result!");
        // 检测到的人脸如果关键点检测置信度过低，标记人脸无效，状态设置为UNKNOWN
        face->faceType = F_TYPE_UNKNOWN;
        if (faceLandmarkTemplateMatchSwitch) {
            _use_template_match = false;
            _template_matrix.release();
        }
        return false;
    }

    return true;
}

/// 扩大 detection 出来的人脸框
void FaceLandmarkDetector::resize_box(int width, int height, FaceInfo *face) {
    // float t_w = face->_rect_rb.x - face->_rect_lt.x;
    // float t_h = face->_rect_rb.y - face->_rect_lt.y;
    float t_max = 0.f;
    //  人脸框的坐标的左上角的进行零值保护
    face->rectLT.x = face->rectLT.x - t_max > 0 ? face->rectLT.x - t_max : 0;
    face->rectLT.y = face->rectLT.y - t_max > 0 ? face->rectLT.y - t_max : 0;
    //  人脸框的坐标的右下角的进行图像的宽高保护
    face->rectRB.x =
            face->rectRB.x + t_max < (float) width ? face->rectRB.x + t_max : (float) width;
    face->rectRB.y =
            face->rectRB.y + t_max < (float) height ? face->rectRB.y + t_max : (float) height;
}

void FaceLandmarkDetector::match_template(VisionRequest *request, FaceInfo *face_info) {
    VTensor img(request->width, request->height, request->frame, INT8);

    // 重新设置原图片的大小
    VTensor resize_img;
    int resize_w = request->width / _k_template_img_scale;
    int resize_h = request->height / _k_template_img_scale;
    va_cv::resize(img, resize_img, va_cv::VSize(resize_w, resize_h), 0, 0, va_cv::INTER_NEAREST);

    // 缩放
    VTensor resize_template;
    va_cv::resize(_template_matrix, resize_template,
                  va_cv::VSize(_template_matrix.w / 8, _template_matrix.h / 8), 0, 0,
                  va_cv::INTER_NEAREST);

    // 模版匹配
    VTensor matched;
    va_cv::match_template(resize_img, resize_template, matched, va_cv::TM_CCOEFF_NORMED);

    //寻找最佳匹配位置
    int max_loc[2];
    va_cv::minMaxIdx(matched, nullptr, nullptr, nullptr, max_loc);

    // 左上角的点
    VPoint top_left(static_cast<float>(max_loc[1] * _k_template_img_scale),
                    static_cast<float>(max_loc[0] * _k_template_img_scale));

    // 右下角的点
    VPoint bottom_right(top_left.x + static_cast<float>(_template_matrix.w),
                        top_left.y + static_cast<float>(_template_matrix.h));

    // 更新人脸框的点
    face_info->rectLT.copy(top_left);
    face_info->rectRB.copy(bottom_right);
}

void FaceLandmarkDetector::get_face_template(VisionRequest *request, FaceInfo *face) {
    VTensor img(request->width, request->height, request->frame, INT8);
    float face_w = face->rectRB.x - face->rectLT.x;
    float face_h = face->rectRB.y - face->rectLT.y;
    float size = MAX(face_w, face_h);
    float pad_size = MIN(20.f, size * 0.1f);
    float new_size = size + pad_size;
    float left = (face->rectLT.x + face->rectRB.x) / 2.f - new_size / 2.f;
    float top = (face->rectLT.y + face->rectRB.y) / 2.f - new_size / 2.f;

    // 裁剪人脸
    left = CLAMP(left, 0, request->width);
    top = CLAMP(top, 0, request->height);
    auto right = CLAMP(left + size, 0, request->width);
    auto bottom = CLAMP(top + size, 0, request->height);
    VRect face_rect(left, top, right, bottom);
    va_cv::crop(img, _template_matrix, face_rect);
    _use_template_match = true; // 下帧数据使用模版匹配
}

} // namespace vision
