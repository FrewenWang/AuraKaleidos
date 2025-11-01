#pragma once

#include "AbsDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include <map>
#include <algorithm>
#include <math.h>

namespace aura::vision {

using RectCenter = std::vector<float>;
using ScoredRect = std::pair<std::pair<VRect, RectCenter>, float>;

class FaceBoxLayer {
public:
    ~FaceBoxLayer();

    FaceBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
                 int num_lvls, std::vector<float> &bias, float nms_thr_outer);

    int Forward(TensorArray &data,
                int det_w, int det_h, int img_w, int img_h,
                std::vector<int> &grid_ws, std::vector<int> &grid_hs,
                std::vector<int> &grid_ss, std::vector<std::vector<float>> &box_rslt);

private:
    std::vector<int> NMS(std::vector<std::pair<float, int> > &scoreIndex,
                         std::vector<std::vector<float>> &bboxes) const;

    std::vector<float> DecodeBBox(const float *pred, int idx, int stride, int lvl_idx, int grid_x, int grid_y,
                                  int input_w, int input_h, int img_w, int img_h, int an_idx,
                                  std::vector<int> &grid_ws, std::vector<int> &grid_hs);

    static float CalConfScore(const float *pred, int idx, int stride, int cls_id);

    static float Sigmoid(float x);

    int num_cls;
    int num_lvls;
    int num_bias;
    std::map<int, float> conf_thr_table;
    float nms_thr;
    float nms_thr_outer;
    std::vector<float> bias;
};

class FaceRectDetector : public AbsDetector<FaceInfo> {
public:
    FaceRectDetector();

    ~FaceRectDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

private:
    void init_params();

    /**
     * @brief 初始化模型输入尺寸与输入图片尺寸缩放比，用于将输入图片缩放到模型输入的大小
     *
     * @param img_w 输入图片的宽
     * @param img_h 输入图片的高
     */
    void compute_output_scale(int img_w, int img_h);

    // 模型图像输入宽度、高度。目前模型：w: 320 h: 224
    int inputWidth;
    int inputHeight;

    // 最近一帧输入图像的宽度、高度
    int lastImgWidth;
    int lastImgHeight;

    // 进行检测的图片的宽高吃土
    int imgWidth;
    int imgHeight;
    // NMS 阈值
    const float nmsThreshold = 0.33f;

    /**
     * 模型检测输出的人脸框的结果信息
     */
    std::vector<std::vector<float>> boxResult;
    /**
     * 设置人脸最多检测的数量：5人
     * 业务设置的需要检测的人脸数目
     */
    short faceMaxCount = 5;
    short faceNeedCheckCount = 0;
    /** result中第几个faceInfo */
    short faceInfoIndex = 0;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped{};
    VTensor tTensorResized{};
};

} // namespace vision
