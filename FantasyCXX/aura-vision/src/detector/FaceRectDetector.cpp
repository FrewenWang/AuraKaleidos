#include "FaceRectDetector.h"

#include <algorithm>

#include "inference/InferenceRegistry.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include "util/TensorConverter.h"
#include "vision/util/VaAllocator.h"
#include "vacv/cv.h"
#include "util/id_util.h"

namespace aura::vision {


static int num_cls = 1;            // 检测类别
static int num_lvls = 3;           // 输出特征图的个数
static int num_bias = 3;           // 偏置个数
static float nms_thr = 0.5f;       // 类内nms的阈值
static float nms_thr_outer = 0.5f; // 类间nms的阈值-目前只检测手势，没有用到这个参数
static std::vector<int> grid_hs{7, 14, 28};   // 三个输出特征图尺寸
static std::vector<int> grid_ws{10, 20, 40};
static std::vector<int> grid_ss{1, 1, 1};
static std::vector<float> bias{220.0f, 125.0f, 128.0f, 222.0f, 264.0f, 266.0f,
                               35.0f, 87.0f, 102.0f, 96.0f, 60.0f, 170.0f,
                               10.0f, 15.0f, 24.0f, 36.0f, 72.0f, 42.0f};
static std::map<int, float> conf_thr_table{{0, 0.2f},
                                           {1, 0.2f},
                                           {2, 0.2f}};
// 非极大值抑制
static FaceBoxLayer face_box(conf_thr_table, nms_thr, num_cls, num_bias,
                             num_lvls, bias, nms_thr_outer);

FaceBoxLayer::FaceBoxLayer(std::map<int, float> conf_thr, float nms_thr, int num_cls, int num_bias,
                           int num_lvls, std::vector<float> &bias, float nms_thr_outer) :
        num_cls(num_cls), num_lvls(num_lvls), num_bias(num_bias), conf_thr_table(conf_thr), nms_thr(nms_thr),
        nms_thr_outer(nms_thr_outer), bias(bias) {
}

FaceBoxLayer::~FaceBoxLayer() = default;

int FaceBoxLayer::Forward(TensorArray &data, int det_w, int det_h, int img_w, int img_h,
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
//        int indsize = indices.size();
        for (int indice: indices) {
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
FaceBoxLayer::DecodeBBox(const float *pred, int idx, int stride, int lvl_idx, int grid_x, int grid_y,
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

float FaceBoxLayer::CalConfScore(const float *pred, int idx, int stride, int cls_id) {
    float objectness1 = (pred[idx * 6 + 4]);
    float confidence1 = (pred[idx * 6 + 4 + (1 + cls_id) * int(stride)]);
    float objectness = Sigmoid(objectness1);
    float confidence = Sigmoid(confidence1);
    return objectness * confidence;
}

float FaceBoxLayer::Sigmoid(float x) {
    return 1. / (1. + exp(-x));
}

std::vector<int>
FaceBoxLayer::NMS(std::vector<std::pair<float, int> > &scoreIndex, std::vector<std::vector<float>> &bboxes) const {
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
        for (int keptIdx: indices) {
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

FaceRectDetector::FaceRectDetector()
        : inputWidth(0),
          inputHeight(0),
          lastImgWidth(0),
          lastImgHeight(0) {
    TAG = "FaceRectDetector";
    mPerfTag += TAG;
}

int FaceRectDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    // 实例化_predictor
    MAKE_PREDICTOR(_predictor, cfg->sourceId, ModelId::VISION_TYPE_FACE_RECT);
    V_CHECK_NULL_RET_INFO(_predictor, Error::PREDICTOR_NULL_ERR, "Face rect predictor not registered!");
    // 通过predictor获取模型信息
    auto model_input_list = _predictor->get_input_desc();
    if (!model_input_list.empty()) {
        auto shape = model_input_list[0].shape();
        inputWidth = shape.w();
        inputHeight = shape.h();
    }

    // init_params() 必须在 init() 最后执行，依赖上面的值。
    init_params();

    V_RET(Error::OK);
}

FaceRectDetector::~FaceRectDetector() {}

int FaceRectDetector::doDetect(VisionRequest *request, VisionResult *result) {
    TensorArray prepared;
    TensorArray predicted;
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pre");
        V_CHECK(prepare(request, result->getFaceResult()->faceInfos, prepared));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pro");
        V_CHECK(process(request, prepared, predicted));
    }
    {
        PERF_AUTO(result->getPerfUtil(), mPerfTag + "-pos");
        V_CHECK(post(request, predicted, result->getFaceResult()->faceInfos));
    }
    V_RET(Error::OK);
}

int FaceRectDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_CHECK_COND_ERR((inputHeight == 0 || inputWidth == 0), Error::MODEL_INIT_ERR, "FaceRect Not Load Model!");
    // init params
    if (request->width != lastImgWidth || request->height != lastImgHeight) {
        VLOGE(TAG, "frame size error frame(%d,%d) lastImage(%d,%d)",
              request->width, request->height, lastImgWidth, lastImgHeight);
        init_params();
    }

    // convert color
    DBG_PRINT_ARRAY((char *) request->getFrame(), 50, "face_rect_prepare_before");
    request->convertFrameToGray();
    DBG_PRINT_ARRAY((char *) request->gray.data, 50, "face_rect_cvt_color_after");

    // resize and no normalize
    va_cv::resizeNoNormalize(request->gray, tTensorResized, {inputWidth, inputHeight});
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_rect_resize_no_normalize_after");

    // 调试逻辑：存储人脸框前处理数据、打印前处理数据、存储前处理图片
    DBG_PRINT_ARRAY((float *) tTensorResized.data, 50, "face_rect_prepare_after");
    DBG_IMG("face_rect_prepare", TensorConverter::convert_to<cv::Mat>(tTensorResized));
    // 调试逻辑：读取前处理之后的RAW数据
    // DBG_READ_RAW("./debug_save/face_rect_prepare.bin", tTensorResized.data, tTensorResized.len());

    // put data
    prepared.clear();
    prepared.emplace_back(tTensorResized);
    V_RET(Error::OK);
}

int FaceRectDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_CHECK(_predictor->predict(inputs, outputs, mPerfUtil));
    V_RET(Error::OK);
}

int FaceRectDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **faces) {
    V_CHECK_COND_ERR(infer_results.size() != 3, Error::INFER_ERR, "Face rect infer results size error");

    for (auto &result: infer_results) {
        if (result.dLayout == NCHW) {
            result = result.changeLayout(NHWC);
        }
        // 调试逻辑：存储后处理RAW数据 用于模型接入测试，不作为正式逻辑
        // DBG_PRINT_ARRAY((char *) result.data, 20, std::string("face_rect_post_") + std::to_string(0));
        // DBG_RAW(std::string("face_rect_post_") + std::to_string(0), TensorConverter::convert_to<cv::Mat>(result));
        // 调试逻辑：读取后处理RAW数据 用于模型接入测试，不作为正式逻辑
        // DBG_READ_RAW(std::string("./debug_save/face_rect_post_") + std::to_string(i) + std::string(".bin"),
        //              infer_results[i].data, infer_results[i].len());
    }

    // 获取原图大小
    imgWidth = (int) mRtConfig->frameWidth;
    imgHeight = (int) mRtConfig->frameHeight;

    boxResult.clear();
    face_box.Forward(infer_results, inputWidth, inputHeight, imgWidth, imgHeight,
                     grid_ws, grid_hs, grid_ss, boxResult);

    if (static_cast<int>(boxResult.size()) <= 0) {
        VLOGD(TAG, "face rect detect: no face !!!");
        for (int i = 0; i < static_cast<int>(mRtConfig->faceMaxCount); ++i) {
            auto *face = faces[i];
            face->clearAll();
        }
        V_RET(Error::NO_FACE);
    }

    // 根据业务设置的需要检测的人脸数。减少性能需要设置业务设置的检测人脸数
    faceMaxCount = V_TO_SHORT(mRtConfig->faceMaxCount);
    faceNeedCheckCount = V_TO_SHORT(mRtConfig->faceNeedDetectCount);
    auto nms_size = V_TO_INT(boxResult.size());
    // 最多只取faceMaxCount(5)数量个人脸框
    nms_size = nms_size <= faceMaxCount ? nms_size : faceMaxCount;
    // 重置faceInfo索引，保证每次模型检测后从result中的第一个faceInfo开始赋值
    faceInfoIndex = 0;

    // 遍历所有模型检测出来的人脸框信息，存储到临时人脸信息目录中，根据距离主驾区域ROI position 由近到远进行排序
    for (int j = 0; j < nms_size; ++j) {
        std::vector<float> &rslt = boxResult[j];
        float rect_confidence = rslt[1];
        if (rect_confidence < nmsThreshold) {
            VLOGD(TAG, "face detect failed !! conf=%f,nmsThreshold=%f", rect_confidence, nmsThreshold);
            continue;
        }
        auto ltX = rslt[2] * imgWidth;
        auto ltY = rslt[3] * imgHeight;
        auto rbX = rslt[4] * imgWidth;
        auto rbY = rslt[5] * imgHeight;

        // 按照算法要求，人脸框中人脸宽度小于100的值需要过滤掉的逻辑暂时注释
        // 移到关键点中进行检测，可以结合landmark的pitch角度进行检测。
        // float minDistance = std::min(rbX - ltX, rbY - ltY);
        // if (minDistance < mRtConfig->faceRectMinPixelThreshold) {
        //     VLOGD(TAG, "face bounding box too small !! minPixel=%f, threshold=%f", minDistance, mRtConfig->faceRectMinPixelThreshold);
        //     continue;
        // }

        // 将模型推理结果填充到FaceInfo中
        auto *face = faces[faceInfoIndex];
        face->id = faceInfoIndex;
        face->faceType = FaceDetectType::F_TYPE_DETECT;
        face->rectConfidence = rect_confidence;
        face->faceRect.set(ltX, ltY, rbX, rbY);
        face->rectLT.x = ltX;
        face->rectLT.y = ltY;
        face->rectRB.x = rbX;
        face->rectRB.y = rbY;
        VLOGD(TAG, "face_rect detect face index:[%s],confidence:[%f],rect=[%f, %f, %f, %f]",
              std::to_string(face->id).c_str(), face->rectConfidence,
              face->rectLT.x, face->rectLT.y, face->rectRB.x, face->rectRB.y);
        ++faceInfoIndex;
    }
    V_RET(Error::OK);
}

void FaceRectDetector::init_params() {
    compute_output_scale((int) mRtConfig->frameWidth, (int) mRtConfig->frameHeight);
}


void FaceRectDetector::compute_output_scale(int img_w, int img_h) {
    lastImgWidth = img_w;
    lastImgHeight = img_h;
}

} // namespace vision
