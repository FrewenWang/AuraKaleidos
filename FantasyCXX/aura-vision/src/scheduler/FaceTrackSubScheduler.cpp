#include "FaceTrackSubScheduler.h"

#include "opencv2/opencv.hpp"
#include "util/id_util.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceTrackSubScheduler";

FaceTemplate::FaceTemplate(int id, bool is_template)
        : id(id), isTemplate(is_template), vRect(0, 0, 0, 0) {

}

FaceTemplate::FaceTemplate(int id, bool is_template, float l, float t, float r, float b)
        : id(id), isTemplate(is_template), vRect(l, t, r, b) {

}

void FaceTemplate::updateRect(VRect &&rect) {
    vRect = rect;
}

bool FaceTemplate::iou(const VRect &input) {
    float intersect_l = input.left > vRect.left ? input.left : vRect.left;
    float intersect_r = input.right < vRect.right ? input.right : vRect.left;
    float intersect_t = input.top > vRect.top ? input.top : vRect.top;
    float intersect_b = input.bottom < vRect.bottom ? input.bottom : vRect.bottom;

    float intersect_w = intersect_r - intersect_l < 0 ? 0 : intersect_r - intersect_l;
    float intersect_h = intersect_b - intersect_t < 0 ? 0 : intersect_b - intersect_t;
    float intersect_area = intersect_h * intersect_w;
    if (intersect_area <= 0) {
        return false;
    }

    float rect_area = (vRect.right - vRect.left) * (vRect.bottom - vRect.top);
    float input_area = (input.right - input.left) * (input.bottom - input.top);
    return (intersect_area / (rect_area + input_area - intersect_area)) > 0.8F;
}

void FaceTrackSubScheduler::run(VisionRequest *request, VisionResult *result) {
    // 如果进行人脸跟踪逻辑的判断，需要先判断人脸框和关键点的能力是否开启.如果能力都没有开启，则直接返回
    if ((!mRtConfig->get_switch(ABILITY_FACE_RECT)) && (!mRtConfig->get_switch(ABILITY_FACE_LANDMARK))) {
        return;
    }
    auto method = V_TO_SHORT(mRtConfig->faceDetectMethod);
    // 默认人脸检测的策略是按照项目配置设置，如果是BENCHMARK_TEST模式，则需要强制设置为DETECT
    if (mRtConfig->releaseMode == BENCHMARK_TEST) {
        method = DETECT;
    }
    if (method != detectMethod) {
        detectMethod = method;
        VLOGD(TAG, "face_rect detect method: %d", detectMethod);
    }
    switch (detectMethod) {
        case FaceDetectMethod::DETECT: {
            modelDetect(request, result);
            break;
        }
        case FaceDetectMethod::TRACK: {
            // 人脸跟随策略不适合多人脸的跟随。会出现人脸丢失或者增加之后，跟随的人脸框不会实时更新
            // 如果需要多人脸检测。目前暂时使用模型检测策略
            faceTrackDetect(request, result);
            break;
        }
        case FaceDetectMethod::MATCH_TEMPLATE: {
            matchTemplate(request, result);
            break;
        }
        default: {
            VLOGE(TAG, "unsupported operation: invalid detect method: %d", method);
        }
    }
}


void FaceTrackSubScheduler::modelDetect(VisionRequest *request, VisionResult *result) {
    runDetectionForcibly(ABILITY_FACE_RECT, request, result);

    {
        PERF_AUTO(PerfUtil::global(), "FaceRectDetector-track")
        // 重置跟踪过程中使用的各种状态量
        mFaceTrackHelper.reset();
        // 如果没有检测到人脸则不进行人脸跟踪
        if (!result->hasFace()) {
            mFaceTrackHelper.faceCopyTrackList(result->getFaceResult()->faceInfos);
            return ;
        }

        // 将检测结果的FaceInfo信息复制到用于跟踪使用的CurFaceInfo集合中，同时计算ROI距离并排序
        if (V_F_TO_BOOL(mRtConfig->useDriverRoiPositionFilter)) {
            mFaceTrackHelper.faceCopyRoiFaceInfo(result);
        } else {
            // 将检测结果的FaceInfo信息复制到用于跟踪使用的CurFaceInfo集合中
            mFaceTrackHelper.faceCopyFaceInfo(result);
        }
        // 如果不是BENCHMARK_TEST模式，则进行人脸跟踪和IOU匹配。否则直接将检测到结果输出
        if (mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
            // 将当前帧检测的人脸与跟踪列表进行匹配，并将跟踪结果更新到FaceInfo中
            mFaceTrackHelper.faceTrackAndMatch(result);
        }
    }

    // 将跟踪匹配后的人脸框送至landmark进行校正
    runDetectionForcibly(ABILITY_FACE_LANDMARK, request, result);

    // 对于landmark校验通过的人脸，将跟踪列表中对应人脸的LostCount置-1
    // 匹配后为track的人脸，保持不变
    // 新加入的人脸只通过Rect检测而没有通过Landmark，将跟踪列表中对应人脸置为无效人脸 F_TYPE_UNKNOWN
    // 匹配成功的人脸在Landmark校验时失败，将其还原为匹配前的状态 F_TYPE_TRACK
    int faceNeedCount = static_cast<int>(mRtConfig->faceNeedDetectCount);
    auto infos = result->getFaceResult()->faceInfos;
    for (int i = 0; i < faceNeedCount; ++i) {
        if (infos[i]->faceType == FaceDetectType::F_TYPE_DETECT) {
            mFaceTrackHelper.trackFacesList[i]->faceLostCount = FaceTrackHelper::DEFAULT_FACE_LOST_COUNT;
        } else if (infos[i]->faceType == FaceDetectType::F_TYPE_TRACK) {
            continue ;
        } else if (mFaceTrackHelper.trackFacesList[i]->faceLostCount > FaceTrackHelper::DEFAULT_FACE_LOST_COUNT) {
            mFaceTrackHelper.trackFacesList[i]->detectType = FaceDetectType::F_TYPE_TRACK;
        } else {
            // 新加入的人脸没通过landmark检测，要将其删除
            mFaceTrackHelper.trackFacesList[i]->detectType = FaceDetectType::F_TYPE_UNKNOWN;
        }
    }
}

void FaceTrackSubScheduler::faceTrackDetect(VisionRequest *request, VisionResult *result) {
    if (V_TO_SHORT(mRtConfig->releaseMode) != ReleaseMode::BENCHMARK_TEST) {
        // 进行人脸跟踪时候判断上一帧结果的人脸数量和需要检测的人脸数量是不是一致。
        // 如果不一致这一帧需要重新进行人脸检测（确保新增加的人脸可以被检测出来）
        bool sameFaceCount = (result->faceCount() == V_TO_SHORT(mRtConfig->faceNeedDetectCount));
        runDetectionIF(ABILITY_FACE_RECT, !sameFaceCount, request, result);
    } else {
        runDetectionForcibly(ABILITY_FACE_RECT, request, result);
    }
    runDetection(ABILITY_FACE_LANDMARK, request, result);
}

void match_template(int width, int height, unsigned char *frame, cv::Mat &faceTemplate,
                    FaceTemplate *pFaceTemplate) {
    // 图像帧数据进行四分之一缩放比例
    // TODO 这个地方有误，这个是设置设置为原来的十六分之一
    float scale = 2;
    // 将YUV转为灰度图 8位无符号的单通道灰度图片
    // TODO 数据转灰度需要使用最新的request的方法
    cv::Mat frame_img(cv::Size(width, height), CV_8UC1, frame);

    // 重新设置原图片的大小(缩放为原来的四分之一)
    cv::Mat resizeFrameImg;
    cv::resize(frame_img, resizeFrameImg, cv::Size(width / scale, height / scale),
               0, 0, cv::INTER_NEAREST);

    // 重新设置模版的大小(缩放为原来的四分之一)
    cv::Mat resize_template_img;
    cv::resize(faceTemplate, resize_template_img, cv::Size(faceTemplate.cols / scale, faceTemplate.rows / scale),
               0, 0, cv::INTER_NEAREST);

    // 模版匹配。进行
    cv::Mat image_matched;
    cv::matchTemplate(resizeFrameImg, resize_template_img, image_matched, cv::TM_CCOEFF_NORMED);

    //寻找最佳匹配位置
    int max_loc[2];
    cv::minMaxIdx(image_matched, nullptr, nullptr, nullptr, max_loc);

    // 左上角的点
    VPoint top_left(max_loc[1] * scale, max_loc[0] * scale);

    // 右下角的点
    VPoint bottom_right(top_left.x + faceTemplate.cols, top_left.y + faceTemplate.rows);

    // 更新模板数据中人脸框的点
    pFaceTemplate->vRect = {top_left.x, top_left.y, bottom_right.x, bottom_right.y};
}

float iou(const VRect &r1, const VRect r2) {
    auto x01 = r1.left;
    auto y01 = r1.top;
    auto x02 = r1.right;
    auto y02 = r1.bottom;
    auto x11 = r2.left;
    auto y11 = r2.top;
    auto x12 = r2.right;
    auto y12 = r2.bottom;
    auto dist_center_x = std::fabs((x01 + x02) / 2.f - (x11 + x12) / 2.f);
    auto dist_center_y = std::fabs((y01 + y02) / 2.f - (y11 + y12) / 2.f);
    auto dist_sum_x = (std::fabs(x01 - x02) + std::fabs(x11 - x12)) / 2.f;
    auto dist_sum_y = (std::fabs(y01 - y02) + std::fabs(y11 - y12)) / 2.f;
    if (dist_center_x > dist_sum_x || dist_center_y > dist_sum_y) {
        return 0.f;
    }
    auto cols = std::min(x02, x12) - std::max(x01, x11);
    auto rows = std::min(y02, y12) - std::max(y01, y11);
    auto intersection = cols * rows;

    auto area1 = (x02 - x01) * (y02 - y01);
    auto area2 = (x12 - x11) * (y12 - y11);
    if ((intersection / area1) > 0.5) {
        return 0.7;
    }
    if ((intersection / area2) > 0.5) {
        return 0.7;
    }

    auto coincide = intersection / (area1 + area2 - intersection);
    return coincide;
}

/**
 * 非极大值抑制，用于多人脸框冗余合并
 * 该版实现取人脸框的并集作为最终人脸框
 * 为保证输出结果的人脸框集合按照置信度排列，输入的人脸框集合需按照置信度事先排序
 * @param needCheckFaceCount 需要检测的人脸数量(最多5张人脸)
 * @param rects              模板匹配出来的人脸
 * @param overlap_thresh     极大值遮挡系数
 * @param no                 TODO 此参数的意义
 * @return
 */
std::vector<FaceTemplate> non_max_suppression(short needCheckFaceCount, const std::vector<FaceTemplate> &rects,
                                              float overlap_thresh, bool no) {
    std::vector<FaceTemplate> nms_res;
    int rect_cnt = rects.size();
    if (rect_cnt <= 0) {
        return nms_res;
    } else if (rect_cnt == 1) {
        nms_res.emplace_back(rects[0]);
        return nms_res;
    }
    // 第一个人脸框置信度最高，首先被选中
    nms_res.emplace_back(rects[0]);

    int nms_cnt = 1;
    // 如果 iou > thresh，则合并两个矩形框，取最大的置信度作为新的矩形框的置信度
    // 如果 iou <= thresh，认为是两个不同的人脸，都保留（最多返回CONFIG._s_face_max_count个人脸框）
    for (int i = 1; i < rect_cnt; ++i) {
        for (int j = 0; j < nms_cnt; ++j) {
            auto &r1 = rects[i].vRect;
            auto &r2 = nms_res[j].vRect;
            if (iou(r1, r2) > overlap_thresh) {
                if (nms_res[j].isTemplate) {
                    if (rects[i].isTemplate) {
                        nms_res[j].id = MAX(nms_res[j].id, rects[i].id);
                    }
                } else {
                    if (rects[i].isTemplate) {
                        nms_res[j].id = rects[i].id;
                        nms_res[j].isTemplate = true;
                    }
                }
//                int id = MAX(nms_res[j].id, rects[i].id);
//                id = nms_res[j].id;
//                if (nms_res[j].id < 4) {
//                    id = rects[i].id;
//                }
//                nms_res[j].id = id;
                if (no) {
                    break;
                }
                auto l = std::min(r1.left, r2.left);
                auto t = std::min(r1.top, r2.top);
                auto r = std::max(r1.right, r2.right);
                auto b = std::max(r1.bottom, r2.bottom);
                nms_res[j].updateRect({l, t, r, b});
                break;
            } else {
                if (j == nms_cnt - 1 && nms_cnt < needCheckFaceCount) {
                    nms_res.emplace_back(rects[i]);
                    nms_cnt++;
                    break;
                }
            }
        }
    }
    return nms_res;
}

void FaceTrackSubScheduler::matchTemplate(VisionRequest *request, VisionResult *result) {
    auto needCheckCount = V_TO_SHORT(mRtConfig->faceNeedDetectCount);

    // 当前模板数量不足需要检测人脸数量时，需要重新检测是否有新的人脸加入
    if (static_cast<int>(faceTemplateList.size()) < needCheckCount) {
        runDetectionForcibly(ABILITY_FACE_RECT, request, result);
    }

    // 获取人脸数据结果
    auto face_list = result->getFaceResult()->faceInfos;
    // 获取图像的 w, h, image
    int width = request->width;
    int height = request->height;
    unsigned char *frame = request->frame;

    // 如果模板为空则执行 detection 检测
    if (faceTemplateList.empty()) {
        for (int i = 0; i < needCheckCount; i++) {
            FaceInfo *face = face_list[i];
            V_CHECK_CONT(face->faceType == FaceDetectType::F_TYPE_UNKNOWN);

            int face_id = FaceIdUtil::instance()->produce();

            faceTemplateList.insert(
                    std::pair<int, FaceTemplate>{face_id, FaceTemplate(face_id, false)});
            face->id = face_id;
            face->stateFaceTracking = F_TRACKING_INIT;
            face->faceType = FaceDetectType::F_TYPE_DETECT;
        }
    } else if (static_cast<int>(faceTemplateList.size()) == needCheckCount) {
        // 如果模板数量和需要检测的数量一致，则不需要重新 detection
        // 清除人脸的所有数据
        for (int i = 0; i < mRtConfig->faceNeedDetectCount; i++) {
            FaceInfo *face = face_list[i];
            face->clearAll();
        }
        // 遍历所有的模板数据，进行模板匹配
        std::vector<FaceTemplate> rects;
        for (auto info: faceTemplateList) {
            // 模板里面的ID为0或者模板数据为空，
            if (info.second.id == 0 || info.second.templateMat.empty()) {
                continue;
            }
            // 挨个模板数据进行模板匹配
            match_template(width, height, frame, info.second.templateMat, &info.second);
            // 将模板匹配的数据放在人脸框集合中
            rects.emplace_back(info.second);
        }
        // 进行人脸框的极大值抑制的计算
        auto nms_rects = non_max_suppression(needCheckCount, rects, 0.3, false);
        // 清除人脸模板的集合
        faceTemplateList.clear();
        //
        for (int i = 0; i < static_cast<int>(nms_rects.size()); ++i) {
            int id = nms_rects[i].id;
            if (!nms_rects[i].isTemplate) {
                id = FaceIdUtil::instance()->produce();
            }
            faceTemplateList.insert(std::pair<int, FaceTemplate>{id,
                                                                 {id, true,
                                                                  nms_rects[i].vRect.left,
                                                                  nms_rects[i].vRect.top,
                                                                  nms_rects[i].vRect.right,
                                                                  nms_rects[i].vRect.bottom}});
        }

        int i = 0;
        for (auto info: faceTemplateList) {
            FaceInfo *face = face_list[i];
            face->id = info.second.id;
            face->rectLT = {info.second.vRect.left, info.second.vRect.top};
            face->rectRB = {info.second.vRect.right, info.second.vRect.bottom};
            face->stateFaceTracking = F_TRACKING_TRACKING;
            face->faceType = FaceDetectType::F_TYPE_DETECT;
            ++i;
        }
    } else {
        // 如果有模板但是数量不足
        std::vector<FaceTemplate> rects;
        for (auto info: faceTemplateList) {
            if (info.second.id == 0 || info.second.templateMat.empty()) {
                continue;
            }

            match_template(width, height, frame, info.second.templateMat, &info.second);
            rects.emplace_back(info.second);
        }

        for (int i = 0; i < mRtConfig->faceNeedDetectCount; i++) {
            FaceInfo *face = face_list[i];
            if (face->faceType == FaceDetectType::F_TYPE_UNKNOWN) {
                break;
            }
            FaceTemplate info(face->id, false, face->rectLT.x, face->rectLT.y, face->rectRB.x, face->rectRB.y);
            rects.emplace_back(info);
        }

        auto nms_rects = non_max_suppression(needCheckCount, rects, 0.3, true);
        faceTemplateList.clear();
        for (int i = 0; i < static_cast<int>(nms_rects.size()); ++i) {
            if (i >= mRtConfig->faceNeedDetectCount) {
                break;
            }

            int id = nms_rects[i].id;
            if (!nms_rects[i].isTemplate) {
                id = FaceIdUtil::instance()->produce();
            }

            faceTemplateList.insert(std::pair<int, FaceTemplate>{id,
                                                                 FaceTemplate(id, nms_rects[i].isTemplate,
                                                                              nms_rects[i].vRect.left,
                                                                              nms_rects[i].vRect.top,
                                                                              nms_rects[i].vRect.right,
                                                                              nms_rects[i].vRect.bottom)});
        }

        for (int i = 0; i < mRtConfig->faceNeedDetectCount; i++) {
            FaceInfo *face = face_list[i];
            face->clearAll();
        }

        int i = 0;
        for (auto info: faceTemplateList) {
            FaceInfo *face = face_list[i];
            face->id = info.second.id;
            face->rectLT = {info.second.vRect.left, info.second.vRect.top};
            face->rectRB = {info.second.vRect.right, info.second.vRect.bottom};
            face->stateFaceTracking = info.second.isTemplate ? F_TRACKING_TRACKING : F_TRACKING_INIT;
            face->faceType = FaceDetectType::F_TYPE_DETECT;
            ++i;
        }
    }

    // 经过模板匹配 || detection 之后检测关键点
    runDetection(ABILITY_FACE_LANDMARK, request, result);

    // 清空模板
    faceTemplateList.clear();

    // 检测关键点识别到的人脸并更新匹配模板。
    for (int i = 0; i < mRtConfig->faceNeedDetectCount; ++i) {
        FaceInfo *face = face_list[i];
        V_CHECK_CONT(face->noFace());

        FaceTemplate faceTemplate(face->id, true);

        // 将YUV转为灰度图
        cv::Mat frame_img(cv::Size(width, height), CV_8UC1, frame);
        float width_temp = (face->rectRB.x - face->rectLT.x) * 1.0;
        float height_temp = (face->rectRB.y - face->rectLT.y) * 1.0;
        float size = MAX(width_temp, height_temp);
        float add_size = MIN(20, size * 0.1f);
        float new_size = size + add_size;
        float x_org = (face->rectLT.x + face->rectRB.x) / 2.0F - new_size * 0.5F;
        float y_org = (face->rectLT.y + face->rectRB.y) / 2.0F - new_size * 0.5F;
        cv::Rect face_rect = cv::Rect(x_org, y_org, new_size, new_size) & cv::Rect(0, 0, width, height);

        // 获取旋转后的嘴部的图像数据
        frame_img(face_rect).copyTo(faceTemplate.templateMat);
        if (faceTemplate.templateMat.empty()) {
            continue;
        }

        // 添加到模板列表中
        faceTemplateList.insert(std::pair<int, FaceTemplate>{face->id, faceTemplate});
    }
}

void FaceTrackSubScheduler::set_config(RtConfig *cfg) {
    mRtConfig = cfg;
    initFaceTrack();
}

void FaceTrackSubScheduler::initFaceTrack() {
    mFaceTrackHelper.init(mRtConfig);
}
}
