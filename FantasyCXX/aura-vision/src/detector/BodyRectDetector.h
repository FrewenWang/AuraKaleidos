#pragma once

#include "AbsBodyDetector.h"
#include "vision/core/bean/BodyInfo.h"
#include <map>
#include <algorithm>
#include <math.h>

namespace aura::vision {

using RectCenter = std::vector<float>;
using ScoredRect = std::pair<std::pair<VRect, RectCenter>, float>;

class BodyRectBoxLayer {
public:
    ~BodyRectBoxLayer();

    BodyRectBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
                     int num_lvls, std::vector<float> &bias, float nms_thr_outer);

    int Forward(TensorArray &data,
                int det_w, int det_h, int img_w, int img_h,
                std::vector<int> &grid_ws, std::vector<int> &grid_hs,
                std::vector<int> &grid_ss, std::vector<std::vector<float>> &box_rslt);

private:
    /**
     * 极大值抑制
     * @param scoreIndex
     * @param bboxes
     * @return
     */
    std::vector<int> NMS(std::vector<std::pair<float, int> > &scoreIndex,
                         std::vector<std::vector<float>> &bboxes) const;

    std::vector<float> DecodeBBox(const float *pred, int idx, int stride, int lvl_idx, int grid_x, int grid_y,
                                  int input_w, int input_h, int img_w, int img_h, int an_idx,
                                  std::vector<int> &grid_ws, std::vector<int> &grid_hs);

    /**
     * 计算置信度分支
     * @param pred
     * @param idx
     * @param stride
     * @param cls_id
     * @return
     */
    static float CalConfScore(const float *pred, int idx, int stride, int cls_id);

    static float Sigmoid(float x);

    int num_cls;
    int num_lvls;
    int num_bias;
    std::map<int, float> conf_thr_table;
    float nms_thr = 0.5f;                   // 类内nms的阈值
    float nms_thr_outer = 0.5f;             // 类间nms的阈值-目前只检测手势，没有用到这个参数
    std::vector<float> bias;
};

class BodyRectDetector : public AbsBodyDetector {
public:
    BodyRectDetector();

    ~BodyRectDetector() override;

    int init(RtConfig *cfg) override;

    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:

    int prepare(VisionRequest *request, BodyInfo** infos, TensorArray& prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, BodyInfo **infos) override;

private:
    void init_params();

    std::vector<ScoredRect> non_max_suppression(const std::vector<ScoredRect> &rects, float overlap_thresh);

    void compute_output_scale(int img_w, int img_h);

    void compute_grid_param(int out_w, int out_h);

    int _input_width;
    int _input_height;
    int _output_width;
    int _output_height;
    int _last_img_width;
    int _last_img_height;

    float _scale_h;
    float _scale_w;
    float _scale_weight;

    // actual body rect area by predict
    float bodyRectArea = 0.0f;

    float *_anchor_center_x;
    float *_anchor_center_y;
    /**
     * HeadShoulder20230908MainQATFP16对应置信度
     * 避免误识别,算法推荐置信度.最新置信度阈值：0.422
     */
    const float _k_nms_thresh = 0.422;
    // abnormal area od body rect, should be smaller than 350000 in case of 1920*1280
    const float absArea = 350000.0f;
    const int _k_output_width;
    const int _k_output_height;
    const int _train_output_grid;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped {};
    VTensor tTensorResized {};
//    VTensor tTensorResizedFp32 {};
};

} // namespace vision
