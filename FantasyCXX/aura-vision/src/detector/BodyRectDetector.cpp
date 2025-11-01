#include "BodyRectDetector.h"

#include <algorithm>

#include "inference/InferenceRegistry.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vision/util/VaAllocator.h"
#include "vacv/cv.h"

using namespace aura::vision;


BodyRectBoxLayer::BodyRectBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
                                   int num_lvls, std::vector<float> &bias, float nms_thr_outer) :
      // 初始化顺序与声明一致
      num_cls(num_cls),num_lvls(num_lvls), num_bias(num_bias), conf_thr_table(conf_thr), nms_thr(nms_thr),
      nms_thr_outer(nms_thr_outer), bias(bias) {
}

BodyRectBoxLayer::~BodyRectBoxLayer() = default;

int BodyRectBoxLayer::Forward(TensorArray &data, int det_w, int det_h, int img_w, int img_h,
                              std::vector<int> &grid_ws, std::vector<int> &grid_hs, std::vector<int> &grid_ss,
                              std::vector<std::vector<float>> &box_rslt) {
    std::vector<std::vector<float>> bboxes;
    std::vector<std::pair<float, int> > scoreIdx;
    for (int c = 0; c < num_cls; c++) {
        bboxes.clear();
        scoreIdx.clear();
        int bboxIdx = 0;
        for (int i = 0; i < num_lvls; i++) {
            const int gridH = grid_hs[i];
            const int gridW = grid_ws[i];
            const int fpStride = grid_ss[i];
//            const int anStride = (num_cls + 5) * fpStride;
            const auto *outData = (const float *) data[i].data;
            for (int yid = 0; yid < gridH; yid++) {
                for (int xid = 0; xid < gridW; xid++) {
                    for (int b = 0; b < num_bias; b++) {
                        int obj_idx = yid * gridW * num_bias + xid * num_bias + b;
                        float score = CalConfScore(outData, obj_idx, fpStride, c);
                        if (score >= conf_thr_table[c]) {
                            auto boxtmps = DecodeBBox(outData, obj_idx, fpStride, i,
                                                      xid, yid, det_w, det_h, img_w, img_h,
                                                      b, grid_ws, grid_hs);
                            boxtmps.emplace_back(score);
                            bboxes.emplace_back(boxtmps);
                            scoreIdx.emplace_back(std::make_pair(score, bboxIdx));
                            bboxIdx++;
                        }
                    }
                }
            }
        }
        std::sort(scoreIdx.begin(), scoreIdx.end(), [](std::pair<float, int> &pair1, std::pair<float, int> &pair2) {
            return pair1.first > pair2.first;
        });
        std::vector<int> indices = NMS(scoreIdx, bboxes);
        for (int indice : indices) {
            std::vector<float> &bb = bboxes[indice];
            float xmin = std::min(std::max(bb[0] / img_w, 0.0f), 1.0f);
            float ymin = std::min(std::max(bb[1] / img_h, 0.0f), 1.0f);
            float xmax = std::min(std::max(bb[2] / img_w, 0.0f), 1.0f);
            float ymax = std::min(std::max(bb[3] / img_h, 0.0f), 1.0f);
            if (xmax <= xmin or ymax <= ymin) {
                continue;
            }
            std::vector<float> result;
            result.emplace_back(c);
            result.emplace_back(bb[4]);
            result.emplace_back(xmin);
            result.emplace_back(ymin);
            result.emplace_back(xmax);
            result.emplace_back(ymax);
            box_rslt.push_back(result);
        }
    }
    return 0;
}

std::vector<float>
BodyRectBoxLayer::DecodeBBox(const float *pred, int idx, int stride, int lvl_idx, int grid_x, int grid_y,
                             int input_w, int input_h, int img_w, int img_h, int an_idx,
                             std::vector<int> &grid_ws, std::vector<int> &grid_hs) {
    int num_bias = 3;
    idx = idx * 6 + 0;
    std::vector<float> box{0.0f, 0.0f, 0.0f, 0.0f};
    float pred_x = pred[idx + 0];
    float pred_y = pred[idx + int(stride)];
    float pred_w = pred[idx + 2 * int(stride)];
    float pred_h = pred[idx + 3 * int(stride)];
    box[0] = (grid_x + Sigmoid(pred_x)) * img_w / grid_ws[lvl_idx];
    box[1] = (grid_y + Sigmoid(pred_y)) * img_h / grid_hs[lvl_idx];
    box[2] = exp(pred_w) * bias[lvl_idx * num_bias * 2 + an_idx * 2] * img_w / input_w;
    box[3] = exp(pred_h) * bias[lvl_idx * num_bias * 2 + an_idx * 2 + 1] * img_h / input_h;
    std::vector<float> box2;
    box2.emplace_back(box[0] - box[2] * 0.5);
    box2.emplace_back(box[1] - box[3] * 0.5);
    box2.emplace_back(box[0] + box[2] * 0.5);
    box2.emplace_back(box[1] + box[3] * 0.5);
    return box2;
}

float BodyRectBoxLayer::CalConfScore(const float *pred, int idx, int stride, int cls_id) {
    float objectness1 = (pred[idx * 6 + 4]);
    float confidence1 = (pred[idx * 6 + 4 + (1 + cls_id) * int(stride)]);
    float objectness = Sigmoid(objectness1);
    float confidence = Sigmoid(confidence1);
    return objectness * confidence;
}

float BodyRectBoxLayer::Sigmoid(float x) {
    return 1. / (1. + exp(-x));
}

std::vector<int>
BodyRectBoxLayer::NMS(std::vector<std::pair<float, int> > &scoreIndex, std::vector<std::vector<float>> &bboxes) const {
    auto computeIoU = [](std::vector<std::vector<float>> &bbox, int idx1, int idx2) -> float {
        std::vector<float> &bb1 = bbox[idx1];
        std::vector<float> &bb2 = bbox[idx2];
        float l1 = bb1[0], t1 = bb1[1], r1 = bb1[2], b1 = bb1[3];
        float l2 = bb2[0], t2 = bb2[1], r2 = bb2[2], b2 = bb2[3];
        float l = std::max(l1, l2);
        float t = std::max(t1, t2);
        float r = std::min(r1, r2);
        float b = std::min(b1, b2);
        if (r <= l or b <= t)
            return 0.0f;
        float area1 = (r1 - l1) * (b1 - t1);
        float area2 = (r2 - l2) * (b2 - t2);
        float area_inter = (r - l) * (b - t);
        float area_iou = area_inter / (area1 + area2 - area_inter);
        return area_iou;
    };
    std::vector<int> indices;
    for (auto i: scoreIndex) {
        int idx = i.second;
        bool keep = true;
        for (int keptIdx : indices) {
            if (keep) {
                float overlap = computeIoU(bboxes, idx, keptIdx);
                keep = overlap <= nms_thr;
            } else {
                break;
            }
        }
        if (keep) {
            indices.emplace_back(idx);
        }
    }
    return indices;
}

BodyRectDetector::BodyRectDetector()
        : _input_width(0),
          _input_height(0),
          _output_width(0),
          _output_height(0),
          _last_img_width(0),
          _last_img_height(0),
          _scale_h(0.f),
          _scale_w(0.f),
          _scale_weight(0.f),
          _anchor_center_x(nullptr),
          _anchor_center_y(nullptr),
          _k_output_width(17),
          _k_output_height(9),
          _train_output_grid(9) {
    TAG = "BodyRectDetector";
    mPerfTag += TAG;
}

int BodyRectDetector::init(RtConfig *cfg) {
	mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_HEAD_SHOULDER);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Head shoulder predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        auto shape = model_input_list[0].shape();
        _input_width = shape.w();
        _input_height = shape.h();
    }

    // init_params() 必须在 init() 最后执行，依赖上面的值。
    init_params();

    V_RET(Error::OK);
}

BodyRectDetector::~BodyRectDetector() {
    if (_anchor_center_x != nullptr) {
        VaAllocator::deallocate(_anchor_center_x);
        _anchor_center_x = nullptr;
    }
    if (_anchor_center_y != nullptr) {
        VaAllocator::deallocate(_anchor_center_y);
        _anchor_center_y = nullptr;
    }
}

int BodyRectDetector::doDetect(VisionRequest *request, VisionResult *result) {
    TensorArray prepared;
    TensorArray predicted;
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
        V_CHECK(prepare(request, result->getBodyResult()->pBodyInfos, prepared));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
        V_CHECK(process(request, prepared, predicted));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
        V_CHECK(post(request, predicted, result->getBodyResult()->pBodyInfos));
    }
    V_RET(Error::OK);
}

int BodyRectDetector::prepare(VisionRequest *request, BodyInfo **infos, TensorArray &prepared) {
    // init params
    if (_input_height == 0 || _input_width == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }
    if (request->width != _last_img_width || request->height != _last_img_height) {
		VLOGE(TAG, "frame size error frame(%d,%d) lastImage(%d,%d)",
			  request->width, request->height, _last_img_width, _last_img_height);
        init_params();
    }
    // convert color
    request->convertFrameToGray();

    // resize and no normalize
    va_cv::resizeNoNormalize(request->gray, tTensorResized, {_input_width, _input_height});
    // put data
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int BodyRectDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Head shoulder predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int BodyRectDetector::post(VisionRequest *request, TensorArray &infer_results, BodyInfo **bodyInfo) {
    V_CHECK_COND(infer_results.empty(), Error::INFER_ERR, "Head shoulder predictor infer results size error");

    // infer result layout change,  nchw -> nhwc
    for (int i = 0; i < static_cast<int>(infer_results.size()); i++) {
        if (infer_results[i].dLayout != NHWC) {
            infer_results[i] = infer_results[i].changeLayout(NHWC);
        }
    }
    int det_w = _input_width;
    int det_h = _input_height;
    // 原图大小
    int img_w = (int) mRtConfig->frameWidth;
    int img_h = (int) mRtConfig->frameHeight;

    static int num_cls = 1;            // 检测类别
    static int num_lvls = 3;           // 输出特征图的个数
    static int num_bias = 3;           // 偏置个数
    static float nms_thr = 0.25f;       // 类内nms的阈值
    static float nms_thr_outer = 0.5f; // 类间nms的阈值-目前只检测手势，没有用到这个参数
    static std::vector<int> grid_hs{6, 12, 24};   // 三个输出特征图尺寸
    static std::vector<int> grid_ws{9, 18, 36};
    static std::vector<int> grid_ss{1, 1, 1};
    static std::vector<float> bias{220.0f, 125.0f, 128.0f, 222.0f, 264.0f, 266.0f,
                            35.0f, 87.0f, 102.0f, 96.0f, 60.0f, 170.0f,
                            10.0f, 15.0f, 24.0f, 36.0f, 72.0f, 42.0f};
    static std::map<int, float> conf_thr_table{{0, 0.2f},
                                               {1, 0.2f},
                                               {2, 0.2f}};
    // 非极大值抑制
    static BodyRectBoxLayer headShoulderBox(conf_thr_table, nms_thr, num_cls, num_bias,
                                            num_lvls, bias, nms_thr_outer);

    std::vector<std::vector<float>> box_rslt;
    headShoulderBox.Forward(infer_results, det_w, det_h, img_w, img_h,
                         grid_ws, grid_hs, grid_ss, box_rslt);

    if (static_cast<int>(box_rslt.size()) <= 0) {
        for (int i = 0; i < V_TO_INT(mRtConfig->bodyMaxCount); ++i) {
            auto *info = bodyInfo[i];
            info->clearAll();
        }
        VLOGD(TAG, "no detect Head shoulder");
        V_RET(Error::NO_HEAD_SHOULDER);
    }

    int max_count = V_TO_INT(mRtConfig->bodyNeedDetectCount);
    int nms_size = box_rslt.size();
    nms_size = nms_size < max_count ? nms_size : max_count;

    for (int i = 0; i < nms_size; ++i) {
        std::vector<float> &rslt = box_rslt[i];
        float rect_confidence = rslt[1];
        if (rect_confidence < _k_nms_thresh) {
            VLOGW(TAG, "head shoulder detect failed conf=%f,nms_thresh=%f", rect_confidence, _k_nms_thresh);
            continue;
        }

        auto ltX = rslt[2] * img_w;
        auto ltY = rslt[3] * img_h;
        auto rbX = rslt[4] * img_w;
        auto rbY = rslt[5] * img_h;
        bodyRectArea = (rbX - ltX) * (rbY - ltY);
        if (bodyRectArea > absArea) {
            VLOGW(TAG, "body detect failed !! bodyRectArea=%f,absArea=%f", bodyRectArea, absArea);
            continue;
        }
        auto *headShoulderInfo = bodyInfo[i];
        headShoulderInfo->rectConfidence = rect_confidence;
        headShoulderInfo->id = i + 1;
        headShoulderInfo->headShoulderRectLT.x = ltX;
        headShoulderInfo->headShoulderRectLT.y = ltY;
        headShoulderInfo->headShoulderRectRB.x = rbX;
        headShoulderInfo->headShoulderRectRB.y = rbY;
        headShoulderInfo->headShoulderRectCenter.x = (headShoulderInfo->headShoulderRectLT.x +
                                                      headShoulderInfo->headShoulderRectRB.x) / 2.0f;
        headShoulderInfo->headShoulderRectCenter.y = (headShoulderInfo->headShoulderRectLT.y +
                                                      headShoulderInfo->headShoulderRectRB.y) / 2.0f;
        VLOGI(TAG, "Original body_rect[%ld] conf=[%f], rect=[%f, %f, %f, %f]",
              headShoulderInfo->id, headShoulderInfo->rectConfidence,
              headShoulderInfo->headShoulderRectLT.x, headShoulderInfo->headShoulderRectLT.y,
              headShoulderInfo->headShoulderRectRB.x, headShoulderInfo->headShoulderRectRB.y);
    }
    V_RET(Error::OK);
}

void BodyRectDetector::init_params() {
    compute_output_scale((int) mRtConfig->frameWidth, (int) mRtConfig->frameHeight);
    compute_grid_param(_k_output_width, _k_output_height);
}

void BodyRectDetector::compute_output_scale(int img_w, int img_h) {
    _last_img_width = img_w;
    _last_img_height = img_h;

    _scale_w = static_cast<float>(_input_width) / static_cast<float>(img_w);
    _scale_h = static_cast<float>(_input_height) / static_cast<float>(img_h);
}

void BodyRectDetector::compute_grid_param(int out_width, int out_height) {
    if (!_anchor_center_x) {
        VaAllocator::deallocate(_anchor_center_x);
    }
    if (!_anchor_center_y) {
        VaAllocator::deallocate(_anchor_center_y);
    }

    _output_width = out_width;
    _output_height = out_height;
    _anchor_center_x = (float *) VaAllocator::allocate(_output_width * 4);
    _anchor_center_y = (float *) VaAllocator::allocate(_output_height * 4);

    float w_h_ratio = static_cast<float>(_input_width) / static_cast<float>(_input_height);
    auto grid_thresh = 0.f;
    if (std::fabs(w_h_ratio) < 1.5f) {
        _scale_weight = 2.0f;
        grid_thresh = 0.f;
    } else {
        _scale_weight = 1.0f;
        grid_thresh = 0.f;
    }

    float cell_offset_x = (static_cast<float>(_input_height) / static_cast<float>(_train_output_grid)) / 2;
    float cell_offset_y = (static_cast<float>(_input_height) / static_cast<float>(_train_output_grid)) / 2;

    float cell_step_x = 0.f;
    if (_output_width <= 1) {
        cell_step_x = static_cast<float>(_input_width) - cell_offset_x * 2.f;
    } else {
        cell_step_x = (static_cast<float>(_input_width) - cell_offset_x * 2.f) / static_cast<float>(_output_width - 1);
    }

    float cell_step_y = 0.f;
    if (_output_height <= 1) {
        cell_step_y = static_cast<float>(_input_height) - cell_step_y * 2.f;
    } else {
        cell_step_y =
                (static_cast<float>(_input_height) - cell_offset_y * 2.f) / static_cast<float>(_output_height - 1);
    }

    int cell_width = static_cast<int>(static_cast<float>(_input_width) / static_cast<float>(_output_width));
    int cell_height = static_cast<int>(static_cast<float>(_input_height) / static_cast<float>(_output_height));

    for (int i = 0; i < _output_width; ++i) {
        _anchor_center_x[i] = cell_offset_x + cell_step_x * i;
    }
    for (int i = 0; i < _output_height; ++i) {
        _anchor_center_y[i] = cell_offset_y + cell_step_y * i;
    }

    // 中心点修正
    if (_output_width % 2 == 1) {
        int center_grid_width = _output_width / 2;
        for (int i = 0; i < center_grid_width; ++i) {
            _anchor_center_x[center_grid_width + i] = _anchor_center_x[center_grid_width + i] -
                                                      grid_thresh * static_cast<float>(i * _input_width);
            _anchor_center_x[center_grid_width - i] = _anchor_center_x[center_grid_width - i] +
                                                      grid_thresh * static_cast<float>(i * _input_width);
        }
    } else {
        int center_grid_width1 = _output_width / 2;
        int center_grid_width2 = _output_width / 2 - 1;
        for (int i = 0; i < center_grid_width2; ++i) {
            _anchor_center_x[center_grid_width1 + i] = _anchor_center_x[center_grid_width1 + i] -
                                                       grid_thresh * static_cast<float>(i * _input_width);
            _anchor_center_x[center_grid_width2 - i] = _anchor_center_x[center_grid_width2 - i] +
                                                       grid_thresh * static_cast<float>(i * _input_width);
        }
    }
}


// 非极大值抑制，用于头肩检测多样性检测冗余合并
// 该版实现取检测框的并集作为最终检测输出
// 为保证输出结果的检测框集合按照置信度排列，输入的检测框集合需按照置信度事先排序
std::vector<ScoredRect> BodyRectDetector::non_max_suppression(const std::vector<ScoredRect> &rects,
                                                              float overlap_thresh) {
    std::vector<ScoredRect> nms_res;
    int rect_cnt = rects.size();
    if (rect_cnt <= 0) {
        return nms_res;
    } else if (rect_cnt == 1) {
        nms_res.emplace_back(rects[0]);
        return nms_res;
    }

    // 第一个检测框置信度最高，首先被选中
    nms_res.emplace_back(rects[0]);
    int nms_cnt = 1;

    // 如果 iou > thresh，则合并两个矩形框，取最大的置信度作为新的矩形框的置信度
    // 如果 iou <= thresh，认为是两个不同的检测输出，都保留（最多返回CONFIG._s_head_shoulder_max_count个检测框）
    for (int i = 1; i < rect_cnt; ++i) {
        for (int j = 0; j < nms_cnt; ++j) {
            auto &r1 = rects[i].first.first;
            auto &r2 = nms_res[j].first.first;

            if (MathUtils::base_iou(r1, r2) > overlap_thresh) {
                auto l = std::min(r1.left, r2.left);
                auto t = std::min(r1.top, r2.top);
                auto r = std::max(r1.right, r2.right);
                auto b = std::max(r1.bottom, r2.bottom);
                nms_res[j] = {{VRect{l, t, r, b}, nms_res[j].first.second}, nms_res[j].second};
                break;
            } else {
                if (j == nms_cnt - 1 && nms_cnt < V_TO_INT(mRtConfig->bodyMaxCount)) {
                    nms_res.emplace_back(rects[i]);
                    nms_cnt++;
                    break;
                }
            }
        }
    }

    return nms_res;
}
