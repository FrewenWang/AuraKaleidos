#include "FaceFeatureDetector.h"

#include "config/static_config/ref_landmark/ref_face_feature.h.in"
#include "inference/InferenceRegistry.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"
#include "vision/util/FaceIdUtil.h"

namespace aura::vision {

FaceFeatureDetector::FaceFeatureDetector()
    : inputWidth(0), inputHeight(0) {
    TAG = "FaceFeatureDetector";
    mPerfTag += TAG;
}

FaceFeatureDetector::~FaceFeatureDetector() = default;

int FaceFeatureDetector::init(RtConfig* cfg) {
    mRtConfig = cfg;
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_FEATURE);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face feature predictor not registered!");

    const auto& model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        const auto& shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }
    // 不需要预设均值和方差
    // init_params();

    V_RET(Error::OK);
}

void FaceFeatureDetector::init_params() {
    //    meanData.create(3, FP32);
    //    stddevData.create(3, FP32);
    //    auto* mean_data = (float*)meanData.data;
    //    auto* stddev_data = (float*)stddevData.data;
    //    for (int i = 0; i < 3; ++i) {
    //        mean_data[i] = 127.5f;
    //        stddev_data[i] = 128.f;
    //    }
}

int FaceFeatureDetector::prepare(VisionRequest *request, FaceInfo** infos, TensorArray& prepared) {
    if (inputHeight == 0 || inputWidth == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    // 按算法要求特殊处理，RGB三个通道填充相同的灰度数据
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *)request->gray.data, 50, "face_feature_prepare_before");

    VTensor sp_rgb(request->width, request->height, 3, INT8, NHWC);
    auto* grey_data = (char*)request->gray.data;
    auto* rgb_data = (char*)sp_rgb.data;
    long size = request->width * request->height;
    for (int i = 0; i < size; ++i) {
        *(rgb_data++) = grey_data[i];
        *(rgb_data++) = grey_data[i];
        *(rgb_data++) = grey_data[i];
    }
    DBG_PRINT_ARRAY((char *)sp_rgb.data, 50, "face_feature_prepare_cvt_rgb");
    DBG_RAW("feature_input_image", TensorConverter::convert_to<cv::Mat>(sp_rgb));

    // warp affine and normalize
    float scale = 1.f;
    float rot = 0.f;
    va_cv::VScalar aux_params;
    const VPoint* landmarks = (*infos)->landmark2D106;
    ImageUtil::get_warp_params(landmarks, get_face_feature_ref_lmk(), scale, rot, aux_params);

    va_cv::warp_affine(sp_rgb, tTensorWarped, scale, rot, va_cv::VSize(inputWidth, inputHeight),
                                 aux_params, va_cv::INTER_LINEAR, va_cv::BORDER_CONSTANT, va_cv::VScalar());

    // resize and normalize 不需要预设均值和方差，算法原始脚本均值和方差是实时计算的
    va_cv::resize_normalize(tTensorWarped, tTensorResized,
                       va_cv::VSize(inputWidth, inputHeight), 0, 0, va_cv::INTER_LINEAR);

    DBG_PRINT_ARRAY((char *)tTensorResized.data, 50, "face_feature_prepare_after");
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceFeatureDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face feature predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceFeatureDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face feature infer result is empty");
    auto& output = infer_results[0];
    auto* out_data = (float*)output.data;
    float* feature = (*infos)->feature;
    // 注意：算法侧原始脚本中会进行归一化操作，工程侧次数不需要进行归一化操作
    // 因为工程侧进行人脸比对的时候会先进行归一化操作
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        feature[i] = out_data[i];
    }

#ifdef DEBUG_SUPPORT
    // 如果开启调试模式之后。每帧会尝试和上一帧的图片进行特征值比对。借此校验特征值比对是否有效
    float score = FaceIdUtil::compare_face_features(preFeature, feature);
    VLOGD(TAG, "compare face feature with preFrame score:%f", score);
    // 将当前帧的特征值进行存储下载。和下一帧可以进行比对
    for (int i = 0; i < FEATURE_COUNT; ++i) {
        preFeature[i] = out_data[i];
    }
#endif

    DBG_PRINT_FACE_FEATURE((*infos));
    V_RET(Error::OK);
}

} // namespace vision