#include "FaceEyeCenterDetector.h"

#include <vector>
#include <math.h>
#include "util/math_utils.h"
#include <numeric>

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/TensorConverter.h"
#include "vision/util/log.h"
#include "vacv/cv.h"
#include "vision/util/MouthUtil.h"
#include "vacv/resize.h"

using namespace aura::vision;

FaceEyeCenterDetector::FaceEyeCenterDetector():
      // 变量初始化顺序需和声明顺序一致
      _ref_face_width(0), _ref_face_height(0),
      _input_width(0), _input_height(0) {
            TAG = "FaceEyeCenterDetector";
            mPerfTag += TAG;
}

FaceEyeCenterDetector::~FaceEyeCenterDetector() = default;

int FaceEyeCenterDetector::init(RtConfig* cfg) {
    mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_EYE_CENTER);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face Eye Center predictor not registered!");

    const auto& model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        const auto& shape = model_input_list[0].shape();
        _input_width = shape.w();
        _input_height = shape.h();
    }

    init_params();

    V_RET(Error::OK);
}

int FaceEyeCenterDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    if (_input_height == 0 || _input_width == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    request->convertFrameToGray();

    VPoint* landmarks = (*infos)->landmark2D106;
    int eye_start = 0;
    int eye_end = 0;

    if (_is_left_eye) {
        eye_start = FLM_51_L_EYE_LEFT_CORNER;
        eye_end = FLM_60_L_EYE_CENTER;
    } else {
        eye_start = FLM_61_R_EYE_LEFT_CORNER;
        eye_end = FLM_70_R_EYE_CENTER;
    }

    VTensor grey_tensor = request->gray;

    /** 
     * 判断SDK是否是 XPERF_TEST，命名风格使用大小写
     */
    if (!useXperfTest) {
        _min_point.x = (float) grey_tensor.w;
        _min_point.y = (float) grey_tensor.h;
        _max_point.x = 0.0;
        _max_point.y = 0.0;

        for (int i = eye_start; i < eye_end; i++) {
            VPoint &p_point = landmarks[i];
            _min_point.x = std::min(_min_point.x, p_point.x);
            _min_point.y = std::min(_min_point.y, p_point.y);

            _max_point.x = std::max(_max_point.x, p_point.x);
            _max_point.y = std::max(_max_point.y, p_point.y);
        }
        /**
         *拓展detect box的检测范围，extendBoxRatio是扩展系数
		 */
        MathUtils::resizeExtendBox(_min_point, _max_point, boxCenterPoint, request->width, request->height, extendBoxRatio);
    } else {
        /**
         * XPERF_TEST 眼球数据集输入为单个人眼图片，因此无需裁剪操作
         */
        _min_point.x = 0.0;
        _min_point.y = 0.0;
        _max_point.x = (float)grey_tensor.w;
        _max_point.y = (float)grey_tensor.h;
    }
    boxWidth = _max_point.x - _min_point.x;
    boxHeight = _max_point.y - _min_point.y;

    VRect rect(_min_point.x, _min_point.y, _min_point.x + boxWidth, _min_point.y + boxHeight);
    if (rect.left < 0 || rect.top < 0 || rect.right > grey_tensor.w || rect.bottom > grey_tensor.h) {
        V_RET(Error::UNKNOWN_FAILURE);
    }

    // crop
    va_cv::crop(grey_tensor, tTensorCropped, rect);
    DBG_PRINT_ARRAY((char *) tTensorCropped.data, 50, "eye_center_crop_after");
    DBG_RAW("eye_center_crop_after", TensorConverter::convert_to<cv::Mat>(tTensorCropped));

#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
    //  resize and no normalize 因为eye center 模型需要将小图 resize 成大图。目前默认的双线形差值fastcv无法和opencv对齐。
    va_cv::resizeNoNormalize(tTensorCropped, tTensorResized, {_input_width, _input_height}, 0, 0,
                            va_cv::VInterMode::INTER_NEAREST);
#else
    //  resize and no normalize 因为eye center 模型需要将小图 resize 成大图。目前默认的双线形差值fastcv无法和opencv对齐。
    va_cv::resizeNoNormalize(tTensorCropped, tTensorResized, {_input_width, _input_height}, 0, 0,
                             va_cv::VInterMode::INTER_LINEAR);
#endif
    
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 20, "eye_center_prepare_after");
    DBG_RAW("eye_center_prepare_after", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    DBG_IMG("eye_center_prepare_after", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    V_RET(Error::OK);
}

int FaceEyeCenterDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs){
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face Eye Center predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceEyeCenterDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face emotion infer result is empty");
    auto& output = infer_results[0];
    auto out_len = output.size();
    auto* output_data = (float*)output.data;
    DBG_PRINT_ARRAY(output_data, out_len, "Face Eye Center infer result");

    if (Logger::getLogLevel() <= LogLevel::DEBUGGER) {
        std::string ds;
        for (int i = 0; i < static_cast<int>(out_len); ++i) {
            ds.append(" ");
            ds.append(std::to_string(output_data[i]));
        }
        DBG_PRINT(TAG, "eye center output data:%s", ds.c_str());
    }

    VPoint* landmarks = (*infos)->landmark2D106;
    auto* face = *infos;

    if (_is_left_eye) {
        if (!useXperfTest) {
            face->eyeCentroidLeft.x = output_data[1] * boxWidth / _input_width + _min_point.x;
            face->eyeCentroidLeft.y = output_data[2] * boxHeight / _input_height + _min_point.y;
            for (int i = 0; i < LM_EYE_2D_8_COUNT; i++) {
                VPoint point(output_data[3 + 2 * i] * boxWidth / _input_width + _min_point.x,
                             output_data[4 + 2 * i] * boxHeight / _input_height + _min_point.y);
                face->eyeLmk8Left[i] = point;
            }
        } else {
            /**
             *XPERF_TEST 只使用左眼检测，其测试集输入为单个人眼图片，此时无需做尺度缩放
             */
            face->eyeCentroidLeft.x = output_data[1];
            face->eyeCentroidLeft.y = output_data[2];
            for (int i = 0; i < 8; i++) {
                VPoint point(output_data[3 + 2 * i], output_data[4 + 2 * i]);
                landmarks[51 + i].copy(point);
                face->eyeLmk8Left[i] = point;
            }
        }

        for (int i = 0; i < LM_EYE_2D_8_COUNT; i++) {
            landmarks[51 + i].copy(face->eyeLmk8Left[i]);
        }

        /**
         *将瞳孔点坐标重新赋给landmark对应的特征点
         */
        landmarks[59].copy(face->eyeCentroidLeft);
        // 使用人脸挂件店左眼上下眼睑距离
        face->eyeEyelidDistanceLeft = powf(powf(landmarks[FLM_58_L_EYE_BOTTOM].x - landmarks[FLM_57_L_EYE_TOP].x, 2) +
                                           powf(landmarks[FLM_58_L_EYE_BOTTOM].y - landmarks[FLM_57_L_EYE_TOP].y, 2), 0.5);
        // 左眼左右眼角的距离。
        face->eyeCanthusDistanceLeft = powf(powf(landmarks[54].x - landmarks[51].x, 2) +
                                            powf(landmarks[54].y - landmarks[51].y, 2), 0.5);
    } else {
        face->eyeCentroidRight.x = output_data[1] * boxWidth / _input_width + _min_point.x;
        face->eyeCentroidRight.y = output_data[2] * boxHeight / _input_height + _min_point.y;

        for (int i = 0; i < LM_EYE_2D_8_COUNT; i++) {
            VPoint point(output_data[3 + 2 * i] * boxWidth / _input_width + _min_point.x,
                         output_data[4 + 2 * i] * boxHeight / _input_height + _min_point.y);
            face->eyeLmk8Right[i] = point;
        }

        for (int i = 0; i < LM_EYE_2D_8_COUNT; i++) {
            landmarks[61 + i].copy(face->eyeLmk8Right[i]);
        }

        /**
         *将瞳孔点坐标重新赋给landmark对应的特征点
         */
        landmarks[69].copy(face->eyeCentroidRight);
        // 右眼上下眼睑距离
        face->eyeEyelidDistanceRight =
                powf(powf(landmarks[FLM_68_R_EYE_BOTTOM].x - landmarks[FLM_67_R_EYE_TOP].x, 2)
                             + powf(landmarks[FLM_68_R_EYE_BOTTOM].y - landmarks[FLM_67_R_EYE_TOP].y, 2),
                     0.5);
        // 右眼左右眼角的距离。
        face->eyeCanthusDistanceRight =
                powf(powf(landmarks[64].x - landmarks[61].x, 2) + powf(landmarks[64].y - landmarks[61].y, 2), 0.5);
    }

    if (_is_left_eye) {
        face->scoreDetectEyeLeftSingle = output_data[19];
        face->scoreDetectPupilLeftSingle = output_data[0];
        face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
    } else {
        face->scoreDetectEyeRightSingle = output_data[19];
        face->scoreDetectPupilRightSingle = output_data[0];
        face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
    }

    // 上下唇距离取100（上嘴唇下）和103（下嘴唇上）之间的距离(像素绝对距离)
    float lipsDistance = MouthUtil::getLipDistance(face);
    // 真实上下嘴唇距离占比人脸框的比例乘以500。求出映射的相对距离
    float mouthOpenDistance = MouthUtil::getLipDistanceRefRect(lipsDistance, std::abs(face->rectLT.y - face->rectRB.y));
    
    if (mRtConfig->sourceId == SOURCE_1) { // dms
        if (_is_left_eye) {
            if (face->scoreDetectEyeLeftSingle < eyeDetectedThresholdDms) {
                face->leftEyeDetectSingle = FaceEyeDetectStatus::EYE_UNAVAILABLE;
                VLOGD(TAG, "eye detected confidence too low!");
            } else if (face->scoreDetectPupilLeftSingle < pupilScoreDms) {
                face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            }
        } else {
            if (face->scoreDetectEyeRightSingle < eyeDetectedThresholdDms) {
                face->rightEyeDetectSingle = FaceEyeDetectStatus::EYE_UNAVAILABLE;
                VLOGD(TAG, "eye detected confidence too low!");
            } else if (face->scoreDetectPupilRightSingle < pupilScoreDms) {
                face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            }
        }
        float yPos = 1300 * (1 - 0.23);
        float meixinPos = yPos - face->landmark2D106[FLM_71_NOSE_BRIDGE1].y;
        if (meixinPos < 50 || mouthOpenDistance > 60) {
            face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
            face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        }
        
        // 跟算法后处理策略顺序，保持一致。将睁眼距离放在逻辑判断最后
        eyeRef = (float) sqrt(pow(face->eyeLmk8Left[6].x - face->eyeLmk8Left[7].x, 2)
                              + pow(face->eyeLmk8Left[6].y - face->eyeLmk8Left[7].y, 2)) * 500 / rectHeight;
        // 为了避免前面的眉心距离和张嘴距离导致某些闭眼被漏检，下面判断睁眼距离。如果小于6.5则依然按照闭眼计算
        if (face->scoreDetectEyeLeftSingle > eyeDetectedThresholdDms
            && face->scoreDetectPupilLeftSingle < pupilScoreDms
            && eyeRef < eyeCloseDistThresholdDms) {
            if (_is_left_eye) {
                face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            } else {
                face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            }
        }
        
        // DMS使用人脸关键点修正过的头姿pitch角度, 如果抬头的角度超过40度，则按照睁眼处理
        if (std::abs(face->optimizedHeadDeflection.pitch) > 40) {
            face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
            face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        }

        // DMS场景下：使用像素差分算法来判断图像是否遮挡，如果图像遮挡则按照睁眼处理
        // 如果没有检测过摄像头遮挡，并且左眼或者右眼不是睁眼的情况下，需要执行摄像头遮挡检测
        if (!hasInvokedCameraCover
            && (face->leftEyeDetectSingle != FaceEyeDetectStatus::PUPIL_AVAILABLE
                || face->rightEyeDetectSingle != FaceEyeDetectStatus::PUPIL_AVAILABLE)) {
            isCameraCovered = checkDmsCameraCover(request->gray, request->width, request->height);
            hasInvokedCameraCover = true;
        }

        // 纯业务策略：哪怕报出来遮挡也不能直接按照不闭眼处理（有部分图片会误报遮挡）
        // 所以哪怕遮挡的图片还是要判断眉心距离
        if (isCameraCovered && meixinPos < 160) {
            face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
            face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        }

    } else if (mRtConfig->sourceId == SOURCE_2) { // oms
        if (_is_left_eye) {
            if (face->scoreDetectEyeLeftSingle < eyeDetectedThresholdOms) {
                face->leftEyeDetectSingle = FaceEyeDetectStatus::EYE_UNAVAILABLE;
                VLOGD(TAG, "eye detected confidence too low!");
            } else if (face->scoreDetectPupilLeftSingle < pupilScoreOms) {
                face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            }
        } else {
            if (face->scoreDetectEyeRightSingle < eyeDetectedThresholdOms) {
                face->rightEyeDetectSingle = FaceEyeDetectStatus::EYE_UNAVAILABLE;
                VLOGD(TAG, "eye detected confidence too low!");
            } else if (face->scoreDetectPupilRightSingle < pupilScoreOms) {
                face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
            }
        }
        // 算法策略：OMS摄像头嘴部张开距离超过40， 则认为是睁眼
        if (mouthOpenDistance > 40) {
            face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
            face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        }
        // 使用人脸关键点修正过的头姿角度判断
        if (std::abs(face->optimizedHeadDeflection.yaw) > 55) {
            face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
            face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        }
    }
    
    
    /// 当前人脸的的情绪如果是Happy、惊讶、生气.并且mouthOpenDistance大约18。
    if ((face->stateEmotionSingle == F_ATTR_EMOTION_HAPPY
         || face->stateEmotionSingle == F_ATTR_EMOTION_SURPRISE
         || face->stateEmotionSingle == F_ATTR_EMOTION_ANGRY) && mouthOpenDistance > 18) {
        face->leftEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
        face->rightEyeDetectSingle = FaceEyeDetectStatus::PUPIL_AVAILABLE;
    }
    
    VLOGD(TAG, "face_eye_center[%ld], _is_left_eye[%d] score_eye[%f, %f] score_pupil[%f, %f] status[%d,%d]", face->id,
          _is_left_eye, face->scoreDetectEyeLeftSingle, face->scoreDetectEyeRightSingle,
          face->scoreDetectPupilLeftSingle, face->scoreDetectPupilRightSingle, face->leftEyeDetectSingle,
          face->rightEyeDetectSingle);

    V_RET(Error::OK);
}

void FaceEyeCenterDetector::init_params() {
}

int FaceEyeCenterDetector::doDetect(VisionRequest *request, VisionResult *result) {
    /// 业务逻辑：开始进行DMS开始进行左右眼检测的时候，重置进行摄像头遮挡检测的标志变量
    if (mRtConfig->sourceId == SOURCE_1) {
        hasInvokedCameraCover = false;
        isCameraCovered = false;
    }
    for (int i = 0; i < static_cast<int>(mRtConfig->faceNeedDetectCount); ++i) {
        auto *face = result->getFaceResult()->faceInfos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出

        // Xperf工具使用的眼球数据集输入为单个人眼图片，所以不需要检测左右眼 并且只使用左眼检测
        // 常规逻辑需要交替检测左右眼
        if (!useXperfTest) {
            _is_left_eye = true;
            doDetectInner(request, result, &face);
            _is_left_eye = false;
            doDetectInner(request, result, &face);
        } else {
            // BENCHMARK_TEST 眼球数据集输入为单个人眼图片，因此不做face id校验，
            // 并且只使用左眼检测
            _is_left_eye = true;
            doDetectInner(request, result, &face);
        }
    }
    V_RET(Error::OK);
}

int FaceEyeCenterDetector::doDetectInner(VisionRequest *request, VisionResult *result, FaceInfo **face) {
    TensorArray prepared;
    TensorArray predicted;
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre")
        V_CHECK_CONT_MSG(prepare(request, face, prepared) != 0, "eye_center prepare error");
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro")
        V_CHECK_CONT_MSG(process(request, prepared, predicted) != 0, "eye_center process error");
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos")
        V_CHECK_CONT_MSG(post(request, predicted, face) != 0, "eye_center post error");
    }
    V_RET(Error::OK);
}

bool FaceEyeCenterDetector::checkDmsCameraCover(const VTensor &src, int w, int h) {
    if (nullptr == src.data || w <= 0 || h <= 0) {
        return false;
    }
    tTensorCoverResized.release();
    imgSrcMat.release();
    imgGaussianBlur.release();
    imgDifference.release();
    ///
    DBG_PRINT_ARRAY((int *)src.data, 20, "checkDmsCameraCover before resize");
    va_cv::Resize::resize(src, tTensorCoverResized, coverDstSize, 0, 0, cv::INTER_AREA);
    DBG_PRINT_ARRAY((int *)tTensorCoverResized.data, 20, "checkDmsCameraCover after resize");
    if (tTensorCoverResized.empty()) {
        VLOGE(TAG, "resizeGray is empty");
        return false;
    }
    // 将据格式如果不是float32可以转化成float32
    tTensorCoverResized = tTensorCoverResized.changeDType(FP32);
    // 将resize之后数据转化成Mat
    imgSrcMat = TensorConverter::convert_to<cv::Mat>(tTensorCoverResized, true);
    DBG_PRINT_ARRAY((float *)imgSrcMat.data, 20, "checkDmsCameraCover after convert_to");
    // 使用3x3的高斯核高斯模糊滤波器对图像进行平滑处理。
    cv::GaussianBlur(imgSrcMat, imgGaussianBlur, gaussianKSize, 1000.0);
    DBG_PRINT_ARRAY((float *)imgGaussianBlur.data, 20, "checkDmsCameraCover after GaussianBlur");
    // 判断图像差异
    imgDifference = cv::abs(imgSrcMat - imgGaussianBlur);

    // block score for multi zones, start from 4(workaround)
    center_block_num = 0;
    int center_inds_up = 6;
    int center_inds_left = 4;
    bool is_block = false;
    for (int i = 0; i < gridNum; ++i) {
        float interval_h_start = i * gridHeight;
        float interval_h_end = (i + 1) * gridHeight;
        for (int j = 0; j < gridNum; ++j) {
            float interval_w_start = j * gridWidth;
            float interval_w_end = (j + 1) * gridWidth;
            // 取一个方形图像
            cv::Mat delta_block = imgDifference(cv::Range(interval_h_start, interval_h_end),
                                                cv::Range(interval_w_start, interval_w_end));
            double mean_delta_block = cv::mean(delta_block)[0];
            if ((i >= center_inds_up) && (j >= center_inds_left)) {
                if (mean_delta_block < dmsThresholdMultiZone) {
                    center_block_num++;
                }
            }
        }
    }
    float block_ratio =
            V_TO_FLOAT(center_block_num) / V_TO_FLOAT((gridNum - center_inds_up) * (gridNum - center_inds_left));
    double meanMatUp = 0.f;
    double meanMatDown = 0.f;
    // # Judge whether block
    if (block_ratio >= dmsThreshGridRatio) {
        is_block = true;
        // 取一个方形图像区域Mat(上方区域)
        cv::Mat matUp = imgSrcMat(cv::Range(0.45 * coverDstSize.h, 0.60 * coverDstSize.h),
                                  cv::Range(0.4 * coverDstSize.w, 0.60 * coverDstSize.w));

        // 取一个方形图像区域Mat(下方方区域)
        cv::Mat matDown = imgSrcMat(cv::Range(0.8 * coverDstSize.h, 0.93 * coverDstSize.h),
                                    cv::Range(0.4 * coverDstSize.w, 0.60 * coverDstSize.w));
        /// 计算上下的
        meanMatUp = cv::mean(matUp)[0];
        meanMatDown = cv::mean(matDown)[0];
        if ((meanMatDown > 100) && (meanMatDown / meanMatUp) > 10) {
            is_block = false;
        }
    } else {
        is_block = false;
    }
    VLOGD(TAG, "face_eye_center is_block[%d], block_ratio[%f], dmsThreshGridRatio[%f], matUp[%lf], matDown[%lf]",
          is_block, block_ratio, dmsThreshGridRatio, meanMatUp, meanMatDown);
    return is_block;
}
