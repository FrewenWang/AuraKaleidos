
#include "AbsFaceDetector.h"

namespace aura::vision {

int AbsFaceDetector::doDetect(VisionRequest *request, VisionResult *result) {
    int ret = V_TO_INT(Error::OK);
    for (int i = 0; i < V_TO_INT(mRtConfig->faceNeedDetectCount); ++i) {
        auto *face = result->getFaceResult()->faceInfos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        TensorArray prepared;
        TensorArray predicted;
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
            ret = prepare(request, &face, prepared);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
            ret = process(request, prepared, predicted);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
            ret = post(request, predicted, &face);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
    }
    result->errorCode = V_TO_SHORT(ret);
    V_RET(ret);
}

} // namespace vision
