#include "FaceAttributeDetector.h"

#include <vector>

#include "config/static_config/ref_landmark/ref_face_attribute.h.in"
#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vacv/cv.h"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"

using namespace aura::vision;

FaceAttributeDetector::FaceAttributeDetector() :
// 变量初始化顺序需要和头文件中定义顺序一致
// boundingBox左上角起点:rectStartX = 10.0f   rectStartY = 10.0f
// boundingBox宽与高: rectWidth = 167.0f rectHeight = 167.0f
        cropRectBox(10.0f, 10.0f, 10.0f + 167.0f, 10.0f + 167.0f),
        inputWidthIR(0), inputHeightIR(0),
        refFaceWidth(0), refFaceHeight(0) {
    TAG = "FaceAttributeDetector";
    mPerfTag += TAG;
}

int FaceAttributeDetector::init(RtConfig* cfg) {
    mRtConfig = cfg;

    initIRParams();

    // 算法最新人脸属性模型不再区分RGB模型和IR模型，暂统一使用IR模型
    //init_rgb_params();

    init_params();

    V_RET(Error::OK);
}

int FaceAttributeDetector::initIRParams() {
    MAKE_PREDICTOR(irPredictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_ATTRIBUTE_IR);
    V_CHECK_NULL_RET_INFO(irPredictor, Error::PREDICTOR_NULL_ERR, "Face Attribute ir predictor not registered!");

    const auto& model_input_list_ir = irPredictor->get_input_desc();
    if (!model_input_list_ir.empty()) {
        const auto& shape = model_input_list_ir[0].shape();
        inputWidthIR = shape.w();
        inputHeightIR = shape.h();
    }

    V_RET(Error::OK);
}

int FaceAttributeDetector::initRgbParams() {
    MAKE_PREDICTOR(rgbPredictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_ATTRIBUTE_RGB);
    V_CHECK_NULL_RET_INFO(rgbPredictor, Error::PREDICTOR_NULL_ERR, "Face Attribute rgb predictor not registered!");

    const auto &model_input_list_rgb = rgbPredictor->get_input_desc();
    if (!model_input_list_rgb.empty()) {
        const auto &shape = model_input_list_rgb[0].shape();
        inputWidthRgb = shape.w();
        inputHeightRgb = shape.h();
    }

    V_RET(Error::OK);
}

void FaceAttributeDetector::init_params() {
    auto* attributeRefLmk = get_attribute_ref_lmk();
    initRefLandmark(attributeRefLmk);
}

int FaceAttributeDetector::prepare(VisionRequest *request, FaceInfo** info, TensorArray& prepared) {
    // init params
    V_CHECK_COND_ERR((inputHeightIR == 0 || inputHeightIR == 0), Error::MODEL_INIT_ERR, "FaceAttribute Not Load Model!");

    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_attribute_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_attribute_convert_color_after");

    // warpAffine 仿射变换
    //if (attTensorWarped.empty()) {
    VTensor warpRotMat;
    const VPoint *landmarks = (*info)->landmark2D106;
    ImageUtil::get_warp_params(landmarks, get_attribute_ref_lmk(), warpRotMat);
    va_cv::warp_affine(request->gray, attTensorWarped, warpRotMat,
                           va_cv::VSize(refFaceWidth, refFaceHeight));
    V_CHECK_COND_ERR(attTensorWarped.w == 0 || attTensorWarped.h == 0, Error::PREPARE_ERR, "FaceAttribute warpAffine result error!");
    //}
    DBG_PRINT_ARRAY((char *) attTensorWarped.data, 50, "face_attribute_warp_affine_after");
    // first resize and no normalize
    va_cv::resizeNoNormalize(attTensorWarped, tTensorFirstResized, {resizeWidth, resizeHeight});
    // crop
    va_cv::crop(tTensorFirstResized, attTensorCropped, cropRectBox);
    DBG_PRINT_ARRAY((char *) attTensorCropped.data, 50, "face_attribute_cropped_after");

    // resize and normalize
    va_cv::resize_normalize(attTensorCropped, tTensorResized, va_cv::VSize(inputWidthIR, inputHeightIR),
                            0, 0, va_cv::INTER_AREA);
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 100, "face_attribute_prepare_after");
    DBG_RAW("face_attribute_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    DBG_IMG("face_attribute_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    V_RET(Error::OK);
}

int FaceAttributeDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(irPredictor, Error::PREDICTOR_NULL_ERR, "Face Attribute ir predictor not registered!");
    V_CHECK(irPredictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceAttributeDetector::post(VisionRequest *request, TensorArray& infer_results, FaceInfo** infos){
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face Attribute infer result is empty");

    auto* faceInfo = *infos;
    auto resGender  = infer_results[0];
    auto resAge     = infer_results[1];
    auto resGlasses = infer_results[2];

    MathUtils::softmax((float*)resAge.data, softmaxAge, 6);
    MathUtils::softmax((float*)resGender.data, softmaxGender, 2);
    MathUtils::softmax((float*)resGlasses.data, softmaxGlass, 3);
    DBG_PRINT_ARRAY((float *) softmaxAge, 6, "face_attribute_age_post");
    DBG_PRINT_ARRAY((float *) softmaxGender, 2, "face_attribute_gender_post");
    DBG_PRINT_ARRAY((float *) softmaxGlass, 3, "face_attribute_glass_post");

    int glass_argmax = MathUtils::argmax((float*)softmaxGlass, 3);
    // 戴眼镜状态映射
    static std::vector<int> glassMap{F_ATTR_GLASS_COLOR, F_ATTR_GLASS_NO_COLOR, F_ATTR_NO_GLASS};
    if (glass_argmax < 3) {
        faceInfo->stateGlassSingle = glassMap[glass_argmax];
    }

    int genderIdx = MathUtils::argmax((float*)softmaxGender, 2);
    // 性别段映射
    static std::vector<int> genderMap{F_ATTR_GENDER_MALE, F_ATTR_GENDER_FEMALE};
    if (genderIdx < 2) {
        faceInfo->stateGenderSingle = genderMap[genderIdx];
    }
    // 年龄段映射
    static std::vector<int> age_map{F_ATTR_AGE_BABY, F_ATTR_AGE_CHILDREN, F_ATTR_AGE_TEENAGER,
                                    F_ATTR_AGE_YOUTH, F_ATTR_AGE_MIDLIFE, F_ATTR_AGE_SENIOR};
    int age_idx = MathUtils::argmax((float *)softmaxAge, 6);
    if (age_idx < 6) {
        faceInfo->stateAgeSingle = age_map[age_idx];
    }
    V_RET(Error::OK);
}

void FaceAttributeDetector::initRefLandmark(VPoint *refLandmarks) {
    if (gBaseLandmarkInitialized) {
        return;
    }
    /// FIXME @wangbin 进行新模型继承时候，和算法同学确认要求Emotion、Attribute、NoInteractLiving、Quality四个模型使用同样的仿射参照系统一起来。
    /*float faceHeight = refLandmarks[16].y -
                       ImageUtil::get_landmarks_y_min(refLandmarks, {34, 35, 36, 43, 44, 45});*/
    float faceWidth = ImageUtil::get_landmarks_x_max(refLandmarks, {32, 31, 30}) -
                      ImageUtil::get_landmarks_x_min(refLandmarks, {0, 1, 2});
    float faceHeight = faceWidth;
    VPoint start(ImageUtil::get_landmarks_x_min(refLandmarks, {0, 1, 2}),
                 ImageUtil::get_landmarks_y_min(refLandmarks, {34, 35, 36, 43, 44, 45}));

    float dx = faceWidth * widthRatio - start.x;
    float dy = faceHeight * heightRatio - start.y;

    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        refLandmarks[i].x =  refRatio * (refLandmarks[i].x + dx);
        refLandmarks[i].y = refRatio * (refLandmarks[i].y + dy);
    }

    refFaceHeight = (int) faceHeight * refRatio * heightChannelRatio;
    refFaceWidth = (int) faceWidth * refRatio * widthChannelRatio;

    gBaseLandmarkInitialized = true;
}