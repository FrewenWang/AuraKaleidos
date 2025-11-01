#include "Source1CameraCoverDetector.h"

#include "inference/InferenceRegistry.h"
#include "util/TensorConverter.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "vacv/cv.h"

namespace aura::vision {

Source1CameraCoverDetector::Source1CameraCoverDetector()
        : inputWidth(0),
          inputHeight(0) {
    TAG = "Source1CameraCoverDetector";
    mPerfTag += TAG;
}

int Source1CameraCoverDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_SOURCE1_CAMERA_COVER);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Camera cover predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        auto shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    init_params();
    V_RET(Error::OK);
}

Source1CameraCoverDetector::~Source1CameraCoverDetector() {

}

int Source1CameraCoverDetector::doDetect(VisionRequest *request, VisionResult *result) {
    TensorArray prepared;
    TensorArray predicted;

    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
        V_CHECK(prepare(request, result->getFaceResult()->faceInfos, prepared));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
        V_CHECK(process(request, prepared, predicted));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
        V_CHECK(post(request, predicted, result->getFaceResult()->faceInfos));
    }

    V_RET(Error::OK);
}

int Source1CameraCoverDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_CHECK_COND_ERR((inputHeight == 0 || inputWidth == 0), Error::MODEL_INIT_ERR, "Not load the model!!!");
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "camera_cover_prepare_before");
    // convert color
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "camera_cover_convert_color_after");

    va_cv::resize_normalize(request->gray, tTensorResized, va_cv::VSize(inputWidth, inputHeight), 0, 0,
                            va_cv::INTER_LINEAR, meanData, stddevData);
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "camera_cover_prepare_after");
    DBG_RAW("camera_cover", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int Source1CameraCoverDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "camera cover predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int Source1CameraCoverDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "camera cover infer result is empty");
    auto &output = infer_results[0];
    auto out_len = output.size();
    auto *out_data = (float *) output.data;
    // 指数归一化
    MathUtils::softmax(out_data, softmaxCallOutput, OUTPUT_DATA_SIZE);
    DBG_PRINT_ARRAY((float *) softmaxCallOutput, out_len, "camera_cover_post_result");
    auto *face = *infos;
    // 获取摄像头单帧结果
    // 指数从小到大排序，取最大索引号
    // model outputs 2 cases: 0：NO_COVER, 1：COVER
    face->scoreCameraCoverSingle = softmaxCallOutput[OUTPUT_DATA_SIZE - 1];
    face->stateCameraCoverSingle = MathUtils::argmax((float *)softmaxCallOutput, out_len);
    V_RET(Error::OK);
}

void Source1CameraCoverDetector::init_params() {
    // resize and normalize
    meanData.create(1, 1, 1, NHWC, FP32);
    stddevData.create(1, 1, 1, NHWC, FP32);
    ((float *) meanData.data)[0] = 127.5f;
    ((float *) stddevData.data)[0] = 128.0f;
}

} // namespace vision
