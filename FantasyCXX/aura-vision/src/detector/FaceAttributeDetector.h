#pragma once

#include <vector>

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class FaceAttributeDetector : public AbsFaceDetector {
public:
    FaceAttributeDetector();

    ~FaceAttributeDetector() {

    }

    int init(RtConfig *cfg) override;

protected:
    /**
     * prepare
     * @param request
     * @param infos
     * @param prepared
     * @return
     */
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    /**
     * process
     * @param request
     * @param inputs
     * @param outputs
     * @return
     */
    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    /**
     * post
     * @param request
     * @param infer_results
     * @param infos
     * @return
     */
    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    /**
     * 初始化人脸属性检测器的参数
     */
    void init_params();

    /**
     * 初始化人脸属性IR模型初始化参数
     * @return
     */
    int initIRParams();

    /**
     * 初始化人脸属性IR模型初始化参数(后续需要废弃)。
     * @return
     */
    int initRgbParams();

    /**
     * 初始化仿射变换参照图片的关键点
     * @param refLandmarks
     */
    void initRefLandmark(VPoint *refLandmarks);

    std::shared_ptr<AbsPredictor> irPredictor;
    std::shared_ptr<AbsPredictor> rgbPredictor;
    /** 进行输入模型图片裁剪的Rect区域 */
    vision::VRect cropRectBox;
    /** 模型输入大小 **/
    int inputWidthIR;
    int inputHeightIR;
    int inputWidthRgb;
    int inputHeightRgb;
    /** 仿射变换参照图的人脸宽高 **/
    int refFaceWidth;
    int refFaceHeight;
    /** 输入ratio参数 **/
    const float widthRatio = 0.25f;
    const float heightRatio = 0.75f;
    const float widthChannelRatio = 1.5f;
    const float heightChannelRatio = 2.0f;
    const float refRatio = 0.799f;
    const int resizeWidth = 187;
    const int resizeHeight = 187;
    bool gBaseLandmarkInitialized = false;
    /** 人脸属性输出的softmax的结果 */
    float softmaxAge[6];
    float softmaxGlass[3];
    float softmaxGender[2];

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorResized{};
    // first resize tensor
    VTensor tTensorFirstResized{};
    // attribute tensor cropped
    VTensor attTensorCropped{};
    // attribute tensor warped
    VTensor attTensorWarped{};

};
} // namepace vision
