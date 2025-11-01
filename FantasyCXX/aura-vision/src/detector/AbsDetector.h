

#pragma once

#include <memory>

#include "inference/AbsPredictor.h"
#include "inference/InferenceRegistry.h"
#include "vision/core/bean/LivingInfo.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/bean/GestureInfo.h"
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VMacro.h"
#include "vision/core/common/VTensor.h"
#include "vision/core/common/VFrame.h"
#include "vision/core/common/VStructs.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include "vision/util/PerfUtil.h"

namespace aura::vision {

enum VColorMode {
    VA_GREY = 0,
    VA_RGB = 1,
    VA_BGR = VA_RGB,
};

class QuantCalibDataUtil;
class RtConfig;

/**
 * @brief Base class of vision ability detector
 * */
template <typename TDetectedInfo>
class AbsDetector {
public:
    virtual int detect(VisionRequest *request, VisionResult *result) {
#ifdef ENABLE_PERF
        mPerfUtil = result->getPerfUtil();
        PERF_AUTO(mPerfUtil, mPerfTag);
#endif
        auto ret = V_TO_INT(Error::UNKNOWN_FAILURE);
        ret = doDetect(request, result);
        result->errorCode = V_TO_SHORT(ret);
        V_RET(ret);
    }

    virtual int doDetect(VisionRequest *request, VisionResult *result) {
		V_RET(Error::OK);
	}
    virtual ~AbsDetector() = default;

    /// init
    virtual int init(RtConfig* cfg) {
        mRtConfig = cfg;
        return 0;
    };

    virtual int deinit() {
        if (_predictor != nullptr) {
            _predictor->deinit();
            _predictor = nullptr;
        }
        mPerfUtil = nullptr;
        mRtConfig = nullptr;
        return 0;
    };

protected:
    /**
     * 注意：模型推理的前处理之后的数据格式是NHWC的
     * 此处不再进行数据帧Layout的转换。
     * 各个predictor根据需要进行对应数据格式转换。
     * 例如：ONNX需要NCHW的、QNN需要NHWC的
     * Prepare the input tensors for inference
     * @param request
     * @param infos
     * @param prepared
     * @return
     */
    virtual int prepare(VisionRequest *request, TDetectedInfo** infos, TensorArray& prepared) = 0;

    /// Inference process
    virtual int process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) = 0;

    /// Post process of the predicted results
    virtual int post(VisionRequest *request, TensorArray& infer_results, TDetectedInfo** infos) = 0;

    std::shared_ptr<AbsPredictor> _predictor;
    PerfUtil* mPerfUtil = nullptr;
    RtConfig* mRtConfig = nullptr;
    char const *TAG = nullptr;
    std::string mPerfTag = "[Detector]: ";

    friend class QuantCalibDataUtil;
};

} // namespace vision
