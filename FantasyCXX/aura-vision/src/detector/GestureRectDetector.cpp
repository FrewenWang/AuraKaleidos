#include "GestureRectDetector.h"

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vision/util/VaAllocator.h"
#include "vacv/cv.h"

namespace aura::vision {

GestureBoxLayer::GestureBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
                                 int num_lvls, std::vector<float> &bias, float nms_thr_outer) :
    num_cls(num_cls),num_lvls(num_lvls),num_bias(num_bias), conf_thr_table(conf_thr), nms_thr(nms_thr),
    nms_thr_outer(nms_thr_outer), bias(bias) {

}

GestureBoxLayer::~GestureBoxLayer() {
}

// output with one cls
int GestureBoxLayer::Forward(TensorArray &data, int det_w, int det_h, int img_w, int img_h,
                             std::vector<int> &grid_ws, std::vector<int> &grid_hs, std::vector<int> &grid_ss,
                             std::vector<std::vector<float>> &box_rslt) {
    std::vector<std::vector<float>> box_internal;
    // assume batchsize is 1
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

            const int anStride = (num_cls + 5) * fpStride;
            const float *outData = (const float *) data[i].data;

            for (int yid = 0; yid < gridH; yid++) {
                for (int xid = 0; xid < gridW; xid++) {
                    for (int b = 0; b < num_bias; b++) {
                        // const int obj_idx = b * anStride + 4 * fpStride + yid * gridW + xid;
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
        /* nms */
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
            box_internal.push_back(result);
        }
    }

    // 类间抑制
    std::vector<std::vector<float>> boxes_hand, boxes_playphone;
    int boxes_hand_len = 0, boxes_playphone_len = 0;
    for(auto item: box_internal){
        if(item[0] == 0){
            boxes_hand.push_back(item);
        }else{
            boxes_playphone.push_back(item);
        }
    }

    boxes_hand_len = boxes_hand.size();
    boxes_playphone_len = boxes_playphone.size();
    std::vector<int> bool_hand(boxes_hand_len, 1);
    std::vector<int> bool_playphone(boxes_playphone_len, 1);

    for(int i = 0; i < boxes_hand_len; i ++){
        if(!bool_hand[i]) continue;
        auto box_item_hand = boxes_hand[i];
        for(int j = 0; j < boxes_playphone_len; j ++){
            if(!bool_playphone[j]) continue;
            auto box_item_playphone = boxes_playphone[j];
            float label1 = box_item_hand[0];
            float score1 = box_item_hand[1];
            float l1 = box_item_hand[2];
            float t1 = box_item_hand[3];
            float r1 = box_item_hand[4];
            float b1 = box_item_hand[5];

            float label2 = box_item_playphone[0];
            float score2 = box_item_playphone[1];
            float l2 = box_item_playphone[2];
            float t2 = box_item_playphone[3];
            float r2 = box_item_playphone[4];
            float b2 = box_item_playphone[5];

            float l = std::max(l1, l2);
            float t = std::max(t1, t2);
            float r = std::min(r1, r2);
            float b = std::min(b1, b2);

            float area_iou = 0.f;
            if(r > l && b > t){
                float area1 = (r1 - l1) * (b1 - t1);
                float area2 = (r2 - l2) * (b2 - t2);
                float area_inter = (r - l) * (b - t);
                area_iou = area_inter / (area1 + area2 - area_inter);
            }
            if(area_iou > 0.5){
                if(score1 > score2)
                    bool_playphone[j] = 0;
                else
                    bool_hand[i] = 0;
            }
        }
    }

    for(int id = 0; id < boxes_hand_len; id ++) {
        if (bool_hand[id]) {
            box_rslt.push_back(boxes_hand[id]);
        }
    }

    for(int id = 0; id < boxes_playphone_len; id ++) {
        if (bool_playphone[id]) {
            box_rslt.push_back(boxes_playphone[id]);
        }
    }

    return 0;
}


std::vector<float>
GestureBoxLayer::DecodeBBox(const float *pred, int idx, int stride, int lvl_idx, int grid_x, int grid_y,
                            int input_w, int input_h, int img_w, int img_h, int an_idx,
                            std::vector<int> &grid_ws, std::vector<int> &grid_hs) {

    int num_bias = 3;
    idx = idx * 7 + 0;
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

float GestureBoxLayer::CalConfScore(const float *pred, int idx, int stride, int cls_id) {
    float objectness1 = (pred[idx * 7 + 4]);
    float confidence1 = (pred[idx * 7 + 4 + (1 + cls_id) * int(stride)]);
    float objectness = Sigmoid(objectness1);
    float confidence = Sigmoid(confidence1);
    return objectness * confidence;
}

float GestureBoxLayer::Sigmoid(float x) {
    return 1. / (1. + exp(-x));
}

std::vector<int>
GestureBoxLayer::NMS(std::vector<std::pair<float, int> > &scoreIndex, std::vector<std::vector<float>> &bboxes) const {
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
    for (auto i : scoreIndex) {
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


GestureRectDetector::GestureRectDetector()
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
      _cell_size(0),
      _train_output_grid(9) {
            TAG = "GestureRectDetector";
            mPerfTag += TAG;
}

int GestureRectDetector::init(RtConfig* cfg) {
	mRtConfig = cfg;

    MAKE_PREDICTOR(_predictor, mRtConfig->sourceId, ModelId::VISION_TYPE_GESTURE_RECT);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Gesture rect predictor not registered!");

    auto model_input_list = _predictor->get_input_desc();
    if (model_input_list.size() > 0) {
        auto shape = model_input_list[0].shape();
        _input_width = shape.w();
        _input_height = shape.h();
    }

    init_params();
    V_RET(Error::OK);
}

GestureRectDetector::~GestureRectDetector() {
    if (_anchor_center_x != nullptr) {
        VaAllocator::deallocate(_anchor_center_x);
        _anchor_center_x = nullptr;
    }
    if (_anchor_center_y != nullptr) {
        VaAllocator::deallocate(_anchor_center_y);
        _anchor_center_y = nullptr;
    }
}

int GestureRectDetector::doDetect(VisionRequest *request, VisionResult *result) {
    if (_input_height == 0 || _input_width == 0) {
        VLOGE(TAG, "Not load the model!!!");
        V_RET(Error::MODEL_INIT_ERR);
    }

    TensorArray prepared;
    TensorArray predicted;
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
        V_CHECK(prepare(request, result->getGestureResult()->gestureInfos, prepared));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
        V_CHECK(process(request, prepared, predicted));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
        V_CHECK(post(request, predicted, result->getGestureResult()->gestureInfos));
    }
    V_RET(Error::OK);
}

int GestureRectDetector::prepare(VisionRequest *request, GestureInfo **infos, TensorArray &prepared) {
    request->convertFrameToGray();
    // VTensor tTensorResized;
    // resize without normalize
    // va_cv::resize(request->gray, resized, {_input_width, _input_height});
    va_cv::resizeNoNormalize(request->gray, tTensorResized, {_input_width, _input_height});

    // 经过前处理之后的Tensor数据是NHWC的
    // 此处不再进行数据帧Layout的转换。各个predictor根据需要进行对应数据格式转换
    // tTensorResized = tTensorResized.changeLayout(NCHW);

    DBG_PRINT_ARRAY((float *) tTensorResized.data, 100, "gest_norm");
    DBG_RAW("gest_norm", TensorConverter::convert_to<cv::Mat>(tTensorResized));

    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int GestureRectDetector::process(VisionRequest *request, TensorArray& inputs, TensorArray& outputs) {
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Gesture rect predictor not registered!");
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int GestureRectDetector::post(VisionRequest *request, TensorArray& infer_results, GestureInfo** infos) {
    V_CHECK_COND(infer_results.size() != 3, Error::INFER_ERR, "Gesture rect infer results size error");

    // infer result layout change,  nchw -> nhwc
    for(int i = 0; i < static_cast<int>(infer_results.size()); i ++){
        if(infer_results[i].dLayout == NCHW){
            infer_results[i] = infer_results[i].changeLayout(NHWC);
        }
    }

    // 模型输入图片大小
    int det_w = _input_width;
    int det_h = _input_height;
    // 原图大小
    int img_w = (int) mRtConfig->frameWidth;
    int img_h = (int) mRtConfig->frameHeight;

    static int num_cls = 2;            // 检测类别， 目前只有手势一种
    static int num_lvls = 3;           // 输出特征图的个数
    static int num_bias = 3;           // 偏置个数
    static float nms_thr = 0.5f;       // 类内nms的阈值
    static float nms_thr_outer = 0.5f; // 类间nms的阈值-目前只检测手势，没有用到这个参数
    static std::vector<int> grid_hs {5, 10, 20};   // 三个输出特征图尺寸
    static std::vector<int> grid_ws {8, 16, 32};
    static std::vector<int> grid_ss {1, 1, 1};
    // 固定的偏置参数
    static std::vector<float> bias  { 220.0f, 125.0f, 128.0f, 222.0f, 264.0f, 266.0f,
                               35.0f, 87.0f, 102.0f, 96.0f, 60.0f, 170.0f,
                               10.0f, 15.0f, 24.0f, 36.0f, 72.0f, 42.0f};
    static std::map<int, float> conf_thr_table { {0, 0.43f}, {1, 0.3f}, {2, 0.2f}};

    GestureBoxLayer gesture_box(conf_thr_table, nms_thr, num_cls, num_bias,
                               num_lvls, bias, nms_thr_outer);

    std::vector<std::vector<float>> box_rslt;
    gesture_box.Forward(infer_results, det_w, det_h, img_w, img_h,
                        grid_ws, grid_hs, grid_ss, box_rslt);

    // 输出检测结果
    if (static_cast<int>(box_rslt.size()) <= 0) {
        for (int i = 0; i < V_TO_INT(mRtConfig->gestureMaxCount); ++i) {
            auto* gesture = infos[i];
            gesture->clear_all();
        }
        V_RET(Error::NO_GESTURE);
    }

    int max_count = V_TO_INT(mRtConfig->gestureNeedDetectCount);
    // 检测到的手势框或者玩手机的方框个数
    int nms_size = box_rslt.size();
    nms_size = nms_size < max_count ? nms_size : max_count;

    for (int i = 0; i < nms_size; ++i) {
        std::vector<float> &rslt = box_rslt[i];
        float rect_confidence = rslt[1];
        int type = static_cast<int>(rslt[0]);
        auto *gesture = infos[i];
        if ((type == _gesture_rect_case && rect_confidence > gestureThresh)) {
            //二分类，检测到的是手，玩手机状态赋值
            gesture->statePlayPhoneSingle = G_PLAY_PHONE_STATUS_NONE;
            gesture->rectConfidence = rect_confidence;
            gesture->id = i + 1;
            gesture->rectLT.x = rslt[2] * img_w;
            gesture->rectLT.y = rslt[3] * img_h;
            gesture->rectRB.x = rslt[4] * img_w;
            gesture->rectRB.y = rslt[5] * img_h;
            gesture->rectType = RectType::G_RECT_TYPE_GESTURE;
            VLOGI(TAG, "gesture_rect[%d] type=[gesture],playPhoneSingle:%d conf=%f, rect=[%f, %f, %f, %f]",
                  gesture->id,
                  gesture->statePlayPhoneSingle, gesture->rectConfidence, gesture->rectLT.x, gesture->rectLT.y,
                  gesture->rectRB.x, gesture->rectRB.y);
        } else if (type == _play_phone_case && rect_confidence > playPhoneThresh) {
            gesture->statePlayPhoneSingle = G_PLAY_PHONE_STATUS_PLAYING;
            gesture->id = i + 1;
            gesture->rectLT.x = rslt[2] * img_w;
            gesture->rectLT.y = rslt[3] * img_h;
            gesture->rectRB.x = rslt[4] * img_w;
            gesture->rectRB.y = rslt[5] * img_h;
            gesture->rectType = RectType::G_RECT_TYPE_PLAY_PHONE;
            VLOGI(TAG, "gesture_rect[%d] type=[play_phone],playPhoneSingle:%d, rect=[%f, %f, %f, %f]",
                  gesture->id,gesture->statePlayPhoneSingle, gesture->rectLT.x, gesture->rectLT.y, gesture->rectRB.x,
                  gesture->rectRB.y);
        } else {
            //二分类，当不满足以上检测判断条件，玩手机状态赋值
            gesture->id = 0.f;
            gesture->rectLT.clear();
            gesture->rectRB.clear();
            gesture->statePlayPhoneSingle = G_PLAY_PHONE_STATUS_NONE;
            gesture->rectType = RectType::G_RECT_TYPE_UNKNOWN;
            VLOGI(TAG, "gesture_rect[%d] type=[other],playPhoneSingle:%d", gesture->id, gesture->statePlayPhoneSingle);
        }
    }
    V_RET(Error::OK);
}

void GestureRectDetector::init_params() {
    compute_output_scale((int) mRtConfig->frameWidth, mRtConfig->frameHeight);
    compute_grid_param(_k_output_width, _k_output_height);
}

void GestureRectDetector::compute_output_scale(int img_w, int img_h) {
    _last_img_width = img_w;
    _last_img_height = img_h;

    _scale_w = static_cast<float>(_input_width) / static_cast<float>(img_w);
    _scale_h = static_cast<float>(_input_height) / static_cast<float>(img_h);
}

void GestureRectDetector::compute_grid_param(int out_width, int out_height) {
    if (!_anchor_center_x) {
        VaAllocator::deallocate(_anchor_center_x);
    }
    if (!_anchor_center_y) {
        VaAllocator::deallocate(_anchor_center_y);
    }

    _output_width = out_width;
    _output_height = out_height;
    _anchor_center_x = (float*)VaAllocator::allocate(_output_width * 4);
    _anchor_center_y = (float*)VaAllocator::allocate(_output_height * 4);

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
        cell_step_y = (static_cast<float>(_input_height) - cell_offset_y * 2.f) / static_cast<float>(_output_height - 1);
    }

    int cell_width = static_cast<int>(static_cast<float>(_input_width) / static_cast<float>(_output_width));
    int cell_height = static_cast<int>(static_cast<float>(_input_height) / static_cast<float>(_output_height));
    _cell_size = cell_width > cell_height ? cell_width : cell_height;

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

// 非极大值抑制，用于多手势框冗余合并（20191218）
std::vector<GestureInfo> GestureRectDetector::non_max_suppression(std::vector<VRect> &rects, float overlap_thresh, std::vector<int> &conf) {
    std::vector<GestureInfo> nms_res;
    int rect_cnt = rects.size();
    if (rect_cnt <= 0) {
        return nms_res;
    } else if (rect_cnt == 1) {
        GestureInfo gestinfo;
        gestinfo.rectConfidence = conf[0];
        gestinfo.rectLT.x = rects[0].left;
        gestinfo.rectLT.y = rects[0].top;
        gestinfo.rectRB.x = rects[0].right;
        gestinfo.rectRB.y = rects[0].bottom;
        nms_res.emplace_back(gestinfo);
        return nms_res;
    }

    std::vector<float> areas;
    for (const auto r : rects) {
        float area = std::fabs((r.right - r.left + 1) * (r.bottom - r.top + 1));
        areas.emplace_back(area);
    }

    auto argsort = [](std::vector<int> list) -> std::vector<int> {
        std::vector<int> res;
        for (int i = 0; i < (int) list.size(); ++i) {
            res.emplace_back(i);
        }

        std::sort(res.begin(), res.end(), [&](int i1, int i2) {
            return list[i1] > list[i2];
        });
        return res;
    };

    std::vector<int> config_index = argsort(conf);

    // 手势框按照交并比分组
    std::vector<std::vector<int>> suppress_group;

    auto intersect = [](const VRect &r1, const VRect &r2) -> VRect {
        auto left = std::max(r1.left, r2.left);
        auto top = std::max(r1.top, r2.top);
        auto right = std::min(r1.right, r2.right);
        auto bottom = std::min(r1.bottom, r2.bottom);
        return VRect{left, top, right, bottom};
    };

    for (int i = 0; i < static_cast<int>(config_index.size()) - 1; ++i) {
        std::vector<int> item{config_index[i]};
        suppress_group.emplace_back(item);
        for (int pos = i + 1; pos < static_cast<int>(config_index.size()); ++pos) {
            int j = config_index[pos];
            auto inter = intersect(rects[config_index[i]], rects[j]);
            float w = std::max(0.f, inter.right - inter.left + 1);
            float h = std::max(0.f, inter.bottom - inter.top + 1);

            if (areas[j] != 0) {
                float overlap = (w * h) / (areas[config_index[i]] + areas[j] - (w * h));
                if (overlap > overlap_thresh) {
                    suppress_group[i].emplace_back(j);
                }
            }
        }
    }

    // 合并一些重复的分组
    std::vector<int> group_cnt;
    for (int i = 0; i < (int) suppress_group.size(); ++i) {
        group_cnt.emplace_back(suppress_group[i].size());
    }

    std::vector<int> group_indices = argsort(group_cnt);
    std::vector<std::vector<int>> merged_suppress_group;
    std::vector<int> group_suppress_idx;
    for (int i = 0; i < (int) group_indices.size(); ++i) {
        merged_suppress_group.emplace_back(suppress_group[group_indices[i]]);
        for (int pos = i + 1; pos < (int) group_indices.size(); ++pos) {
            int j = group_indices[pos];
            std::vector<int> repeat_items;
            for (int k = 0; k < (int) merged_suppress_group[i].size(); ++k) {
                if (std::find(suppress_group[j].begin(), suppress_group[j].end(), merged_suppress_group[i][k])
                    != suppress_group[j].end()) {
                    repeat_items.emplace_back(k);
                    break;
                }
            }
            if ((int) repeat_items.size() != 0) {
                for (int k = 0; k < (int) suppress_group[j].size(); ++k) {
                    if (std::find(merged_suppress_group[i].begin(), merged_suppress_group[i].end(),
                                  suppress_group[j][k])
                        == merged_suppress_group[i].end()) {
                        merged_suppress_group[i].emplace_back(suppress_group[j][k]);
                    }
                }
                group_suppress_idx.emplace_back(pos);
            }
        }

        int tmp = 0;
        for (int sidx : group_suppress_idx) {
            group_indices.erase(group_indices.begin() + sidx - tmp);
            tmp++;
        }
        group_suppress_idx.clear();

    }

    //求同一分组最大置信度的矩形框
    std::vector<int> pick_idx;
    for (int i = 0; i < (int) merged_suppress_group.size(); ++i) {
        int max_confi = 0;
        int max_confi_Index = -1;

        for (int j = 1; j < (int) merged_suppress_group[i].size(); ++j) {
            if (conf[merged_suppress_group[i][j]] > max_confi) {
                max_confi = conf[merged_suppress_group[i][j]];
                max_confi_Index = merged_suppress_group[i][j];
            }
        }
        if (max_confi_Index >= 0) {
            pick_idx.emplace_back(max_confi_Index);
        }
    }

    for (int i = 0; i < static_cast<int>(pick_idx.size()); ++i) {
        GestureInfo gest_info;
        gest_info.rectConfidence = conf[pick_idx[i]];
        gest_info.rectLT.x = rects[pick_idx[i]].left;
        gest_info.rectLT.y = rects[pick_idx[i]].top;
        gest_info.rectRB.x = rects[pick_idx[i]].right;
        gest_info.rectRB.y = rects[pick_idx[i]].bottom;

        nms_res.emplace_back(gest_info);
    }
    return nms_res;
}

} // namespace vision
