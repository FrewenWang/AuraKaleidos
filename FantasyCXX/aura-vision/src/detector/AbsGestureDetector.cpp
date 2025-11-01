

#include "AbsGestureDetector.h"

namespace aura::vision {

int AbsGestureDetector::doDetect(VisionRequest *request, VisionResult *result) {
    int ret = V_TO_INT(Error::OK);
    for (int i = 0; i < static_cast<int>(mRtConfig->gestureNeedDetectCount); ++i) {
        auto *info = result->getGestureResult()->gestureInfos[i];
        V_CHECK_CONT(static_cast<int>(info->id) == 0);
        TensorArray prepared;
        TensorArray predicted;
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
            ret = prepare(request, &info, prepared);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
            ret = process(request, prepared, predicted);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
        {
            PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
            ret = post(request, predicted, &info);
            V_CHECK_CONT(ret != V_TO_INT(Error::OK));
        }
    }
    result->errorCode = V_TO_SHORT(ret);
    V_RET(ret);
}

} // namespace vision
