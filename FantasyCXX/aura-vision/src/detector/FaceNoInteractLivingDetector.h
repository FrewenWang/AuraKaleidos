#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"

namespace aura::vision {

class FaceLivenessDetector : public AbsFaceDetector {
public:
    FaceLivenessDetector();
    ~FaceLivenessDetector() override;

    int init(RtConfig* cfg) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;
    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;
    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    int init_ir_params();

    int init_rgb_params();

    //后处理归一化数组的大小,liviness/attack两组数据
    static const int OUTPUT_DATA_SIZE = 2;

    std::shared_ptr<AbsPredictor> _ir_predictor;
    std::shared_ptr<AbsPredictor> _rgb_predictor;

    int _ref_face_width;
    int _ref_face_height;
    int _input_width_ir;
    int _input_height_ir;
    int _input_width_rgb;
    int _input_height_rgb;

    const int _k_rect_start_x = 22;
    const int _k_rect_start_y = 22;
    const int _k_rect_width = 202;
    const int _k_rect_height = 202;

    //Rect的左上角与右下角坐标
    vision::VRect rectBox;

    // 输出的归一化数组，取最大元素下标
    float softmaxLivinessOutput[OUTPUT_DATA_SIZE];
    // 均值方差
    VTensor omsMean;
    VTensor omsStddev;
    VTensor dmsMean;
    VTensor dmsStddev;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorIrResized {};

    VTensor tTensorRgbRotMat {};
    VTensor tTensorRgbWarpped {};
    VTensor tTensorRgbCropped {};
    VTensor tTensorRgbResized {};
//    VTensor tTensorRgbResizedFp32 {};
};

} // namespace vision
