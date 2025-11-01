#include "FaceMouthLandmarkDetector.h"

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "vacv/cv.h"
#include "util/TensorConverter.h"

namespace aura::vision {

FaceMouthLandmarkDetector::FaceMouthLandmarkDetector()
        : mInputWidth(0),
          mInputHeight(0),
          mouthExtendWidth(0.f),
          mouthExtendHeight(0.f) {
    TAG = "FaceMouthLandmarkDetector";
    mPerfTag += TAG;
}

FaceMouthLandmarkDetector::~FaceMouthLandmarkDetector() = default;

int FaceMouthLandmarkDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_FACE_MOUTH_LANDMARK);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face mouth landmark predictor not registered!");
    // 获取模型反序列化信息
    const auto &model_input_list = _predictor->get_input_desc();
    // mouth box clear
    minPoint.clear();
    maxPoint.clear();
    mouthCenter.clear();
    if (!model_input_list.empty()) {
        const auto &shape = model_input_list[0].shape();
        mInputWidth = shape.w();
        mInputHeight = shape.h();
    }
    V_RET(Error::OK);
}

int FaceMouthLandmarkDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_CHECK_COND_ERR((mInputHeight == 0 || mInputWidth == 0), Error::MODEL_INIT_ERR, "MouthLandmark Not Load Model!");
    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_mouth_landmark_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_mouth_landmark_cvt_color_after");
    tensorPrepare = request->gray;
    minPoint.x = (float) tensorPrepare.w;
    minPoint.y = (float) tensorPrepare.h;
    maxPoint.x = 0.0;
    maxPoint.y = 0.0;
    for (int i = FLM_86_MOUTH_LEFT_CORNER; i <= FLM_105_MOUTH_LOWER_LIP_RIGHT_CT2; i++) {
        mouthPoint = (*infos)->landmark2D106[i];
        minPoint.x = std::min(minPoint.x, mouthPoint.x);
        minPoint.y = std::min(minPoint.y, mouthPoint.y);

        maxPoint.x = std::max(maxPoint.x, mouthPoint.x);
        maxPoint.y = std::max(maxPoint.y, mouthPoint.y);
    }
    // 将mouth框缩放为原来的1.2倍
    MathUtils::resizeExtendBox(minPoint, maxPoint, mouthCenter, request->width, request->height, extendBoxRatio);
    mouthExtendWidth = maxPoint.x - minPoint.x;
    mouthExtendHeight = maxPoint.y - minPoint.y;
    // crop算法原始模型使用使用int类型进行裁剪的。所以此处进行转化到int类型
    VRect rect((int) minPoint.x, (int) minPoint.y, (int) (minPoint.x + mouthExtendWidth), (int) (minPoint.y + mouthExtendHeight));
    if (rect.left < 0 || rect.top < 0 || rect.right > tensorPrepare.w || rect.bottom > tensorPrepare.h) {
        V_RET(Error::UNKNOWN_FAILURE);
    }

    va_cv::crop(request->gray, tTensorCropped, rect);
    DBG_PRINT_RECT(rect, "face_mouth_landmark_crop_rect");
    DBG_PRINT_ARRAY((char *) tTensorCropped.data, 50, "face_mouth_landmark_crop_after");

    // resize and no normalize
    va_cv::resizeNoNormalize(tTensorCropped, tTensorResized, {mInputWidth, mInputHeight});

    // 调试逻辑：存储mouth框前处理数据、打印前处理数据、存储前处理图片
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_mouth_landmark_prepare_after");
    DBG_IMG("face_mouth_landmark_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    // 调试逻辑：读取前处理之后的RAW数据
    // DBG_READ_RAW("./debug_save/face_mouth_landmark_prepare.bin", tTensorResized.data, tTensorResized.len());

    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceMouthLandmarkDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face mouth landmark predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceMouthLandmarkDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Face mouth landmark infer result is empty");
    auto &output = infer_results[0];
    auto *out_data = (float *) output.data;
    auto *face = *infos;
    for (int j = 0; j < LM_MOUTH_2D_20_COUNT; j++) {
        face->mouthLmk20[j].x = out_data[2 * j] * mouthExtendWidth / mInputWidth + minPoint.x;
        face->mouthLmk20[j].y = out_data[2 * j + 1] * mouthExtendHeight / mInputHeight + minPoint.y;
    }
    V_RET(Error::OK);
}

} // namespace vision
