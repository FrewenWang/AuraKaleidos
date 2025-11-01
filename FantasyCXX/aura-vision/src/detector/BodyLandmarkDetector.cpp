#include "BodyLandmarkDetector.h"

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "vacv/cv.h"
#include "util/TensorConverter.h"

namespace aura::vision {

BodyLandmarkDetector::BodyLandmarkDetector()
        : mInputWidth(0),
          mInputHeight(0),
          bodyRectWidth(0.f),
          bodyRectHeight(0.f) {
    TAG = "BodyLandmarkDetector";
    mPerfTag += TAG;
}

BodyLandmarkDetector::~BodyLandmarkDetector() = default;

int BodyLandmarkDetector::init(RtConfig* cfg) {
    mRtConfig = cfg;
    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_BODY_LANDMARK);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Body landmark predictor not registered!");

    const auto& model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        const auto& shape = model_input_list[0].shape();
        mInputWidth = shape.w();
		mInputHeight = shape.h();
    }
    V_RET(Error::OK);
}

int BodyLandmarkDetector::prepare(VisionRequest *request, BodyInfo **infos, TensorArray &prepared) {
    if (mInputHeight == 0 || mInputWidth == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    auto *body = *infos;
    // predict body location according to HeadShoulder points
    //predictBodyLocation(body->headShoulderRectLT, body->headShoulderRectRB, body->headShoulderRectCenter, request->width, request->height);

    // 裁剪Body区域
    bodyRectWidth = body->headShoulderRectRB.x - body->headShoulderRectLT.x;
    bodyRectHeight = body->headShoulderRectRB.y - body->headShoulderRectLT.y;
    V_CHECK_COND(bodyRectWidth <= 0 || bodyRectHeight <= 0, Error::PREPARE_ERR, "crop body size invalid!");
    request->convertFrameToGray();
    VRect rect(body->headShoulderRectLT.x, body->headShoulderRectLT.y, body->headShoulderRectRB.x, body->headShoulderRectRB.y);
    if (rect.left < 0 || rect.top < 0 || rect.right > request->gray.w || rect.bottom > request->gray.h) {
        V_RET(Error::UNKNOWN_FAILURE); // todo: better errcode?
    }

    // VTensor tTensorCropped;
    va_cv::crop(request->gray, tTensorCropped, rect);
    // resize and no normalize
    va_cv::resizeNoNormalize(tTensorCropped, tTensorResized, {mInputWidth, mInputHeight});
    // 调试逻辑：存储人脸框前处理数据、打印前处理数据、存储前处理图片
    DBG_PRINT_ARRAY((float *)tTensorResized.data, 20, "body_landmark_prepare");
    DBG_IMG("body_landmark_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    // 调试逻辑：读取前处理之后的RAW数据
    DBG_READ_RAW("./debug_save/body_landmark_prepare.bin", tTensorResized.data, tTensorResized.len());

    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int BodyLandmarkDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Body landmark predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int BodyLandmarkDetector::post(VisionRequest *request, TensorArray &infer_results, BodyInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Body landmark infer result is empty");
    auto &output = infer_results[0];
    auto *out_data = (float *) output.data;
    auto *body = *infos;

    auto maxPointX = static_cast<float>(request->width);
    auto maxPointY = static_cast<float>(request->height);
    // get bodyLandmarkConfidence
    body->bodyLandmarkConfidence = out_data[BODY_LM_2D_12_COUNT * 2];
    if (body->bodyLandmarkConfidence < bodyLandmarkThreshold) {
        VLOGW(TAG, "Body Landmark[%s] conf too low! conf=%f,threshold=%f",
              std::to_string(body->id).c_str(), body->bodyLandmarkConfidence,
              bodyLandmarkThreshold);
        body->clearAll();
        V_RET(Error::BODYLANDMARK_ERR);
    }
    for (int j = 0; j < BODY_LM_2D_12_COUNT; j++) {
        VPoint &p = body->bodyLandmark2D12[j];
        // x:out_data[0] ~ out_data[11], y:out_data[12] ~ out_data[23]
        p.x = out_data[j] * bodyRectWidth / mInputWidth + body->headShoulderRectLT.x;
        p.y = out_data[j + BODY_LM_2D_12_COUNT] * bodyRectHeight / mInputHeight + body->headShoulderRectLT.y;
        if (maxPointX < p.x) p.x = maxPointX;
        if (maxPointY < p.y) p.y = maxPointY;
        if (minPointX > p.x) p.x = minPointX;
        if (minPointY > p.y) p.y = minPointY;
    }
    VLOGI(TAG, "body_landmark[%ld], landmarkConf=%f, resize_rect=[%f, %f, %f, %f]", body->id, body->bodyLandmarkConfidence,
          body->headShoulderRectLT.x, body->headShoulderRectLT.y, body->headShoulderRectRB.x, body->headShoulderRectRB.y);
    V_RET(Error::OK);
}

/// predict body rect location
void BodyLandmarkDetector::predictBodyLocation(VPoint& leftUp, VPoint& rightDown, VPoint& rectCenter, int frameWidth, int frameHeight) {
    float boxWidth = rightDown.x - leftUp.x;
    float boxHeight = rightDown.y - leftUp.y;
    float rectEdge = bodyExtendRatio * std::max(boxWidth, boxHeight);
    rectCenter.x = (rightDown.x + leftUp.x) / 2.0f;
    rectCenter.y = (rightDown.y + leftUp.y) / 2.0f;
    leftUp.x = std::max(0.0f,rectCenter.x - rectWidthRatio * rectEdge);
    leftUp.y = std::max(0.0f,rectCenter.y - rectLTHeightRatio * rectEdge);
    rightDown.x = std::min(rectCenter.x + rectWidthRatio * rectEdge, (float) frameWidth);
    rightDown.y = std::min(rectCenter.y + rectRBHeightRatio * rectEdge, (float) frameHeight);
}

} // namespace vision
