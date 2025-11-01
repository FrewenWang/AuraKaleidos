#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <math.h>

#include "AbsGestureDetector.h"
#include "vision/core/bean/GestureInfo.h"

namespace aura::vision {

class GestureBoxLayer {
public:
    ~GestureBoxLayer();

    GestureBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
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

/**
 * @brief 手势框（Bounding Box）算法能力检测器
 *
 * 负责调用手势框模型执行推理计算，并负责相关前处理、后处理操作；
 * 人脸框检测算法采用 YOLOv1 算法
 */
class GestureRectDetector : public AbsGestureDetector {
public:
    GestureRectDetector();
    ~GestureRectDetector() override;

    int init(RtConfig* cfg) override;

//    int doDetect(VFrameInfo& frame, GestureInfo** infos, PerfUtil* perf) override;
    int doDetect(VisionRequest *request, VisionResult *result) override;

protected:
//    int prepare(VFrameInfo& frame, GestureInfo** infos, TensorArray& prepared) override;
    int prepare(VisionRequest *request, GestureInfo** infos, TensorArray& prepared) override;
    int process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) override;
    int post(VisionRequest *request, TensorArray& infer_results, GestureInfo** infos) override;

private:
    void init_params();

    /**
     * @brief 初始化模型输入尺寸与输入图片尺寸缩放比，用于将输入图片缩放到模型输入的大小
     *
     * @param img_w 输入图片的宽
     * @param img_h 输入图片的高
     */
    void compute_output_scale(int img_w, int img_h);

    /**
 * @brief 初始化用于计算模型输出的人脸框相关的参数
 *
 * @param out_w
 * @param out_h
 */
    void compute_grid_param(int out_w, int out_h);
    std::vector<GestureInfo> non_max_suppression(std::vector<VRect>& rects, float overlap_thresh, std::vector<int>& conf);

    // 模型图像输入宽度、高度
    int _input_width;
    int _input_height;

    // 模型输出特征图宽度、高度
    int _output_width;
    int _output_height;

    // 最近一帧输入图像的宽度、高度
    int _last_img_width;
    int _last_img_height;

    // 最近一帧输入图像的宽度、高度
    float _scale_h;
    float _scale_w;
    float _scale_weight;

    // 每个手势检测框的中心点坐标（共 9 x 9 个检测框）
    float* _anchor_center_x;
    float* _anchor_center_y;

    // 每个人脸检测框的边长尺寸
    int _cell_size;

    // NMS 阈值
    const float gestureThresh = 0.485f;
    /**
     * HandPhoneDetection1031V17MainNoQATFP16
     * 玩手机的的阈值，最新模型阈值0.51
     */
    const float playPhoneThresh = 0.54;
    const int _gesture_rect_case = 0;
    const int _play_phone_case = 1;

    // 模型输出特征图宽度、高度 初始化指
    const int _k_output_width = 17;
    const int _k_output_height = 9;

    const int _k_filter_size = 80;

    // 算法给出的人脸检测框的宫格维度：9
    const int _train_output_grid;

    // -------------------------------------------------------------------------
    // doDetect() 过程中的临时变量，为避免每帧频繁申请内存，作为成员变量声明
    VTensor tTensorCropped {};
    VTensor tTensorResized {};
//    VTensor tTensorResizedFp32 {};
};

} // namespace vision
