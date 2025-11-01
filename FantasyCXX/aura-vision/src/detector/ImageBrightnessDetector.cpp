
#include "ImageBrightnessDetector.h"
#include "util/DebugUtil.h"

namespace aura::vision {

static const char *TAG = "ImageBrightnessDetector";

int ImageBrightnessDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    V_RET(Error::OK);
}

bool ImageBrightnessDetector::checkOverDark(VTensor &gray) {
    int dark = 0;                          // 偏暗的像素
    float ratio = 0;                       // 偏暗像素所占比例
    for (int i = 0; i < gray.mPixelSize; i++) {
        if (((char *)gray.data)[i] < 40) { // 0~39的灰度值为暗
            dark++;
        }
    }
    ratio = (float)dark / (float)gray.mPixelSize;
    bool overDark = ratio >= OVER_DARK_RATIO_THRESHOLD;
    VLOGD(vision::TAG, "overDark[%d] darkNum[%d] pixelSize[%d] ratio[%f]", overDark, dark, gray.mPixelSize, ratio);
    return overDark;
}

int ImageBrightnessDetector::doDetect(VisionRequest *request, VisionResult *result) {
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "image_brightness_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "image_brightness_convert_color_after");
    auto face = result->getFaceResult()->faceInfos[0];
    bool isOverDark = checkOverDark(request->gray);
    face->stateBrightnessSingle = isOverDark ? ImageBrightness::I_STATE_OVER_DARK : ImageBrightness::I_STATE_NORMAL;

    V_RET(Error::OK);
}

int ImageBrightnessDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_RET(Error::OK);
}

int ImageBrightnessDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_RET(Error::OK);
}

int ImageBrightnessDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_RET(Error::OK);
}

} // namespace vision