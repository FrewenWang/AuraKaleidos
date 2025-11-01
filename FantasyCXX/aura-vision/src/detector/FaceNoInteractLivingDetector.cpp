#include "FaceNoInteractLivingDetector.h"

#include <climits>

#include "config/static_config/ref_landmark/ref_face_liveness.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"

namespace aura::vision {

FaceLivenessDetector::FaceLivenessDetector()
        : _ref_face_width(0), _ref_face_height(0),
          _input_width_ir(0), _input_height_ir(0),
          _input_width_rgb(0), _input_height_rgb(0),
          rectBox(10, 10, 177, 177) {
    TAG = "FaceLivenessDetector";
    mPerfTag += TAG;
}

FaceLivenessDetector::~FaceLivenessDetector() = default;

int FaceLivenessDetector::init(RtConfig *cfg) {
	mRtConfig = cfg;

    init_params();

    init_ir_params();

    init_rgb_params();

    V_RET(Error::OK);
}

int FaceLivenessDetector::init_ir_params() {
    MAKE_PREDICTOR(_ir_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_NO_INTERACT_LIVING_IR);
    V_CHECK_NULL_RET_INFO(_ir_predictor, Error::PREDICTOR_NULL_ERR, "Face liveness ir predictor not registered!");

    const auto &model_input_list_ir = _ir_predictor->get_input_desc();
    if (!model_input_list_ir.empty()) {
        const auto &shape = model_input_list_ir[0].shape();
        _input_width_ir = shape.w();
        _input_height_ir = shape.h();
    }
    dmsMean.create(1, 1, 1, NHWC, FP32);
    dmsStddev.create(1, 1, 1, NHWC, FP32);
    ((float*)dmsMean.data)[0]   = 127.5f;
    ((float*)dmsStddev.data)[0] = 128.0f;
    V_RET(Error::OK);
}

int FaceLivenessDetector::init_rgb_params() {
    MAKE_PREDICTOR(_rgb_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_NO_INTERACT_LIVING_RGB);
    V_CHECK_NULL_RET_INFO(_rgb_predictor, Error::PREDICTOR_NULL_ERR, "Face liveness rgb predictor not registered!");

    const auto &model_input_list_rgb = _rgb_predictor->get_input_desc();
    if (!model_input_list_rgb.empty()) {
        const auto &shape = model_input_list_rgb[0].shape();
        _input_width_rgb = shape.w();
        _input_height_rgb = shape.h();
    }
    omsMean.create(3, FP32);
    omsStddev.create(3, FP32);
    auto* mean_data = (float*)omsMean.data;
    auto* stddev_data = (float*)omsStddev.data;
    for (int i = 0; i < 3; ++i) {
        mean_data[i] = 127.5f;
        stddev_data[i] = 128.f;
    }

    V_RET(Error::OK);
}

int FaceLivenessDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    // 注，RGB 模式和 IR 模式使用了两种不同的求仿射变换矩阵的方法...
    VTensor processed;
    if (mRtConfig->sourceId == SOURCE_1) {
        //only DMS and IR setting will enter this prepare case
        V_CHECK_COND_ERR((_input_height_ir == 0 || _input_height_ir == 0), Error::MODEL_INIT_ERR,
                         "FaceLivenessIR Not Load Model!");
        DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "no_interact_ir_prepare_before");

        // convert color
        request->convertFrameToGray();
        DBG_PRINT_ARRAY((char *) request->gray.data, 50, "no_interact_ir_cvt_color_after");

        // warpAffine 仿射变换
        if ((*infos)->tTensorWarped.empty()) {
            VTensor warpRotMat;
            const VPoint *landmarks = (*infos)->landmark2D106;
            ImageUtil::get_warp_params(landmarks, get_liveness_ir_ref_lmk(), warpRotMat);
            va_cv::warp_affine(request->gray, (*infos)->tTensorWarped, warpRotMat,
                               va_cv::VSize(_ref_face_width, _ref_face_height));
            V_CHECK_COND_ERR((*infos)->tTensorWarped.w == 0 || (*infos)->tTensorWarped.h == 0, Error::PREPARE_ERR,
                             "FaceNoInteractLivingIR warpAffine result error!");
        }
        DBG_PRINT_ARRAY((char *) (*infos)->tTensorWarped.data, 50, "no_interact_ir_warp_affine_after");


        // cropped;
        if ((*infos)->tTensorCropped.empty()) {
            va_cv::crop((*infos)->tTensorWarped, (*infos)->tTensorCropped, rectBox);
        }
        DBG_PRINT_ARRAY((char *) (*infos)->tTensorCropped.data, 50, "no_interact_ir_crop_after");
        DBG_RAW("liveness_ir_cropped_raw", TensorConverter::convert_to<cv::Mat>((*infos)->tTensorCropped));

        // resize and normalize
        va_cv::resize_normalize((*infos)->tTensorCropped, processed, va_cv::VSize(_input_width_ir, _input_height_ir),
                                0, 0, va_cv::INTER_AREA, dmsMean, dmsStddev);
        DBG_PRINT_ARRAY((float *) processed.data, 50, "no_interact_ir_prepare_after");
        DBG_RAW("liveness_ir_processed_raw", TensorConverter::convert_to<cv::Mat>(processed));

    } else if (mRtConfig->sourceId == SOURCE_2) {
        // OMS_TYPE_IR , OMS_TYPE_RGB and mobile phone will enter this case
        V_CHECK_COND_ERR((_input_height_rgb == 0 || _input_width_rgb == 0), Error::MODEL_INIT_ERR,
                         "FaceLivenessRGB Not Load Model!");

        request->convertFrameToBGR();

        DBG_RAW("liveness_rgb_raw", TensorConverter::convert_to<cv::Mat>(request->bgr));
        DBG_PRINT_ARRAY((char *) request->bgr.data, 100, "liveness_rgb");

        // warpAffine 仿射变换
        const VPoint *landmarks = (*infos)->landmark2D106;
        ImageUtil::get_warp_params(landmarks, get_liveness_ir_ref_lmk(), tTensorRgbRotMat);
        DBG_PRINT_ARRAY((float *) tTensorRgbRotMat.data, 6, "no_interact_prepare_rot_mat");

        // VTensor warp_affine
        va_cv::warp_affine(request->bgr, tTensorRgbWarpped, tTensorRgbRotMat, va_cv::VSize(_ref_face_width, _ref_face_height));
        DBG_PRINT_ARRAY((float *) tTensorRgbWarpped.data, 100, "liveness_rgb_norm");
        DBG_RAW("liveness_norm", TensorConverter::convert_to<cv::Mat>(tTensorRgbWarpped));

        // VTensor crop
        va_cv::crop(tTensorRgbWarpped, tTensorRgbCropped, rectBox);
        // VTensor resize_normalize;
        va_cv::resize_normalize(tTensorRgbCropped, processed, va_cv::VSize(_input_width_rgb, _input_height_rgb), 0, 0,
                                va_cv::INTER_AREA, omsMean, omsStddev);

        // 经过前处理之后的Tensor数据是NHWC的
        // 此处不再进行数据帧Layout的转换。各个predictor根据需要进行对应数据格式转换
        // if(_ir_predictor->getSupportedLayout() == NCHW) {
        //     processed = tTensorRgbResized.changeLayout(NCHW);
        // }
        DBG_PRINT_ARRAY((float *) processed.data, 100, "liveness_chw");
        DBG_RAW("liveness_chw", TensorConverter::convert_to<cv::Mat>(tTensorRgbResized));
    } else {
        VLOGE(TAG, "Unknown camera type, error param choice!!");
        V_RET(Error::INVALID_PARAM);
    }

    DBG_PRINT_ARRAY((float *) processed.data, 50, "no_interact_prepare_after");
    DBG_RAW("liveness_input_raw", TensorConverter::convert_to<cv::Mat>(processed));

    prepared.clear();
    prepared.emplace_back(processed);
    V_RET(Error::OK);
}

int FaceLivenessDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    if (mRtConfig->sourceId == SOURCE_1) {
        V_CHECK_NULL_RET_INFO(_ir_predictor, Error::PREDICTOR_NULL_ERR, "Face liveness dms ir predictor not registered!");
        V_CHECK(_ir_predictor->predict(inputs, outputs, mPerfUtil));
    } else {
        V_CHECK_NULL_RET_INFO(_rgb_predictor, Error::PREDICTOR_NULL_ERR, "Face liveness oms rgb predictor not registered!");
        V_CHECK(_rgb_predictor->predict(inputs, outputs, mPerfUtil));
    }
    V_RET(Error::OK);
}

int FaceLivenessDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face liveness infer result is empty");
    float threshold = 0.f;
    // 根据使用的是OMS还是DMS针对性获取置信度
    if (mRtConfig->sourceId == SOURCE_1) {
        threshold = mRtConfig->noInteractiveLivingIrThreshold;
    } else {
        threshold = mRtConfig->noInteractiveLivingRgbThreshold;
    }
    auto &output = infer_results[0];
    auto *out_data = (float *) output.data;
    // VLOGD(TAG, "liveness predict raw output=[%f,%f]", out_data[0], out_data[1]);
    // 指数归一化
    MathUtils::softmax(out_data, softmaxLivinessOutput, OUTPUT_DATA_SIZE);
    float liveness_score = softmaxLivinessOutput[0];
    auto *face = *infos;
    // 设置无感活体分数
    face->scoreNoInteractLiving = liveness_score;
    face->stateNoInteractLivingSingle =
        liveness_score > threshold ? F_NO_INTERACT_LIVING_LIVING : F_NO_INTERACT_LIVING_ATTACK;
    V_RET(Error::OK);
}

void FaceLivenessDetector::init_params() {
    VPoint max_point(0, 0);
    VPoint min_point(INT_MAX, INT_MAX);
    /// FIXME @wangbin 进行新模型继承时候，和算法同学确认要求Emotion、Attribute、NoInteractLiving、Quality四个模型使用同样的仿射参照系统一起来。
    auto *ref_lmk = get_liveness_ir_ref_lmk();
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

    _ref_face_height = static_cast<int>(face_height * 1.2f);
    _ref_face_width = static_cast<int>(face_width * 1.2f);
}

} // namespace vision
