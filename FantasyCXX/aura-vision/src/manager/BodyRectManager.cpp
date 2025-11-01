#include "BodyRectManager.h"
#include "detector/BodyRectDetector.h"
#include "vision/util/log.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "BodyRectManager";

BodyRectStrategy::BodyRectStrategy(RtConfig *cfg) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

BodyRectStrategy::~BodyRectStrategy() = default;

void BodyRectStrategy::setupSlidingWindow() {
}

void BodyRectStrategy::execute(BodyInfo *pBodyInfo) { }

void BodyRectStrategy::execute(VisionRequest *request, FaceInfo *pFaceInfo, BodyInfo *pBodyInfo, int statusCode) {
    // 算法模型检测策略
    int widValue = request->width;
    if (statusCode == FaceBodyStatus::STATUS_NO_FACE_BODY || statusCode == FaceBodyStatus::STATUS_UNKNOWN) {
        VLOGE(TAG, "body_rect[%d] no face and no body");
        return;
    } else if (statusCode == FaceBodyStatus::STATUS_NO_FACE_HAS_BODY) {
        predictBodyLocation(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                            pBodyInfo->headShoulderRectCenter, widValue, request->height, 0.5f, 0.2f, 0.5f, 0.8f, 1.5f);
        if (pBodyInfo->headShoulderRectCenter.x < 0.5f * widValue &&
            pBodyInfo->headShoulderRectRB.x > 0.5f * widValue) {
            pBodyInfo->headShoulderRectRB.x = 0.5f * widValue;
        } else if (pBodyInfo->headShoulderRectCenter.x > 0.5f * widValue &&
                   pBodyInfo->headShoulderRectLT.x < 0.5f * widValue) {
            pBodyInfo->headShoulderRectLT.x = 0.5f * widValue;
        }
    } else if (statusCode == FaceBodyStatus::STATUS_HAS_FACE_NO_BODY) {
        pBodyInfo->headShoulderRectLT = pFaceInfo->rectLT;
        pBodyInfo->headShoulderRectRB = pFaceInfo->rectRB;
        pBodyInfo->headShoulderRectCenter = pFaceInfo->rectCenter;
        bool ratio = (pBodyInfo->headShoulderRectLT.x + pBodyInfo->headShoulderRectRB.x) > widValue;
        if (ratio && pFaceInfo->optimizedHeadDeflection.roll > angleMax) {
            predictBodyLocation(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                                pBodyInfo->headShoulderRectCenter, widValue, request->height, 0.6f, 0.25f, 0.4f, 0.75f,
                                4.0f);
        } else if (!ratio && pFaceInfo->optimizedHeadDeflection.roll < angleMin) {
            predictBodyLocation(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                                pBodyInfo->headShoulderRectCenter, widValue, request->height, 0.4f, 0.25f, 0.6f, 0.75f,
                                4.0f);
        } else {
            predictBodyLocation(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                                pBodyInfo->headShoulderRectCenter, widValue, request->height, 0.5f, 0.25f, 0.5f, 0.75f,
                                4.0f);
        }
        if (ratio && (pBodyInfo->headShoulderRectLT.x < 0.5f * widValue)) {
            pBodyInfo->headShoulderRectLT.x = 0.5f * widValue;
        } else if (!ratio && (pBodyInfo->headShoulderRectRB.x > 0.5f * widValue)) {
            pBodyInfo->headShoulderRectRB.x = 0.5f * widValue;
        }
    } else if (statusCode == STATUS_HAS_FACE_HAS_BODY) {
        widthDelta = pBodyInfo->headShoulderRectRB.x - pBodyInfo->headShoulderRectLT.x;
        heightDelta = pBodyInfo->headShoulderRectRB.y - pBodyInfo->headShoulderRectLT.y;
        shoulderRatio = widthDelta / heightDelta;
        fedge = 1.5f * std::max(widthDelta, heightDelta);
        if ((widthDelta > widthDeltaMaxA && shoulderRatio > widthHeightRatio) || widthDelta > widthDeltaMaxB) {
            fedge = 1.5f * std::min(widthDelta, heightDelta);
        }
        pBodyInfo->headShoulderRectCenter.x =
                0.5f * (pBodyInfo->headShoulderRectRB.x + pBodyInfo->headShoulderRectLT.x);
        pBodyInfo->headShoulderRectCenter.y =
                0.5f * (pBodyInfo->headShoulderRectRB.y + pBodyInfo->headShoulderRectLT.y);
        if (pFaceInfo->optimizedHeadDeflection.roll < angleMin &&
            std::abs(pFaceInfo->optimizedHeadDeflection.pitch) < angleMax) {
            calculateRectBox(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                             pBodyInfo->headShoulderRectCenter, 0.4f, 0.2f, 0.6f, 0.8f, fedge);
        } else if (pFaceInfo->optimizedHeadDeflection.roll > angleMax &&
                   std::abs(pFaceInfo->optimizedHeadDeflection.pitch) < angleMax) {
            calculateRectBox(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                             pBodyInfo->headShoulderRectCenter, 0.6f, 0.2f, 0.4f, 0.8f, fedge);
        } else {
            calculateRectBox(pBodyInfo->headShoulderRectLT, pBodyInfo->headShoulderRectRB,
                             pBodyInfo->headShoulderRectCenter, 0.5f, 0.2f, 0.5f, 0.8f, fedge);
        }
        bool whRatio = (pBodyInfo->headShoulderRectLT.x + pBodyInfo->headShoulderRectRB.x) > widValue;
        if (pBodyInfo->headShoulderRectCenter.x < 0.5f * widValue &&
            pBodyInfo->headShoulderRectRB.x > 0.5f * widValue) {
            pBodyInfo->headShoulderRectRB.x = 0.5f * widValue;
        } else if (pBodyInfo->headShoulderRectCenter.x > 0.5f * widValue &&
                   pBodyInfo->headShoulderRectLT.x < 0.5f * widValue) {
            pBodyInfo->headShoulderRectLT.x = 0.5f * widValue;
        }
    }
    pBodyInfo->headShoulderRectLT.x = std::max(0.0f, pBodyInfo->headShoulderRectLT.x);
    pBodyInfo->headShoulderRectLT.y = std::max(0.0f, pBodyInfo->headShoulderRectLT.y);
    pBodyInfo->headShoulderRectRB.x = std::min((float) widValue, pBodyInfo->headShoulderRectRB.x);
    pBodyInfo->headShoulderRectRB.y = std::min((float) request->height, pBodyInfo->headShoulderRectRB.y);
}

/// predict body rect location
void BodyRectStrategy::predictBodyLocation(VPoint &leftUp, VPoint &rightDown, VPoint &rectCenter, int frameWidth,
                                           int frameHeight, float a, float b, float c, float d, float e) {
    float boxWidth = rightDown.x - leftUp.x;
    float boxHeight = rightDown.y - leftUp.y;
    float rectEdge = e * std::max(boxWidth, boxHeight);
    rectCenter.x = (rightDown.x + leftUp.x) / 2.0f;
    rectCenter.y = (rightDown.y + leftUp.y) / 2.0f;
    leftUp.x = std::max(0.0f, rectCenter.x - a * rectEdge);
    leftUp.y = std::max(0.0f, rectCenter.y - b * rectEdge);
    rightDown.x = std::min(rectCenter.x + c * rectEdge, (float) frameWidth);
    rightDown.y = std::min(rectCenter.y + d * rectEdge, (float) frameHeight);
}

void
BodyRectStrategy::calculateRectBox(VPoint &leftUp, VPoint &rightDown, VPoint &rectCenter, float a, float b, float c,
                                   float d, float fedge) {
    leftUp.x = rectCenter.x - a * fedge;
    leftUp.y = rectCenter.y - b * fedge;
    rightDown.x = rectCenter.x + c * fedge;
    rightDown.y = rectCenter.y + d * fedge;
}

void BodyRectStrategy::clear() {
}


BodyRectManager::BodyRectManager() {
    detector = std::make_shared<BodyRectDetector>();
}

void BodyRectManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
    bodyRectStrategy = std::make_shared<BodyRectStrategy>(cfg);
}

void BodyRectManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
    bodyRectStrategy = nullptr;
}

bool BodyRectManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_BODY_HEAD_SHOULDER);
}

void BodyRectManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_BODY_HEAD_SHOULDER);

    detector->detect(request, result);

    useFaceRectList.clear();
    useBodyRectList.clear();
    bodyFaceBothRectList.clear();

    auto bodyNeedCount = V_TO_INT(mRtConfig->bodyNeedDetectCount);
    auto faceNeedCount = V_TO_INT(mRtConfig->faceNeedDetectCount);

    for (int i = 0; i < faceNeedCount; ++i) {
        auto face = result->getFaceResult()->faceInfos[i];
        // 遍历所有人脸，如果是无效人脸就没有必要进行下面的计算
        if (!face->hasFace()) {
            continue;
        }
        int matchCount = 0;
        for (int j = 0; j < bodyNeedCount; ++j) {
            auto body = result->getBodyResult()->pBodyInfos[j];
            // 如果没有有效肢体框则，没有必要进行下面的计算
            if (!body->hasBody()) {
                continue;
            }
            auto iouScore = getMatchIOU(face, body);
            if (iouScore > iouThreshold) {
                bodyFaceBothRectList.emplace_back(j, i);
                body->hasMatchedFace = true;
                ++matchCount;
            }
        }
        // 如果当前人脸，没有任何一个肢体框能够与之匹配。则将此人脸的索引值加入人脸框列表
        if (matchCount == 0) {
            useFaceRectList.emplace_back(i);
        }
    }

    // 遍历所有的肢体框,如果对应的肢体框没有匹配成功的人脸，则将其加入到肢体框列表
    for (int i = 0; i < bodyNeedCount; ++i) {
        auto body = result->getBodyResult()->pBodyInfos[i];
        if (!body->hasBody()) {
            continue;
        }
        if (!body->hasMatchedFace) {
            useBodyRectList.emplace_back(i);
        }
    }

    bodyCount = 0;
    // 首先, 遍历同时检测出肢体框和人脸框同时检测出来
    auto bodyFaceBothSize = bodyFaceBothRectList.size();
    for (int i = 0; i < bodyFaceBothSize; ++i) {
        auto data = bodyFaceBothRectList.at(i);
        auto body = result->getBodyResult()->pBodyInfos[data.first];
        auto face = result->getFaceResult()->faceInfos[data.second];
        bodyCount++;
        VLOGD(TAG, "status_has_face_has_body executed: body old[%d]->new[%d]  faceId[%d]", body->id, bodyCount,face->id);
        // 因为判断是否有肢体的逻辑是判断 id > 0,所以body的值从1开始
        body->id = bodyCount;
        // 由于bodyCount是从0开始，所以可以直接使用bodyCount-1作为matchIndex的值
        body->matchIndex = bodyCount - 1;
        bodyRectStrategy->execute(request, face, body, FaceBodyStatus::STATUS_HAS_FACE_HAS_BODY);
    }

    // 其次。遍历计算检测出来肢体框，但是没有检测出人脸框
    auto useBodySize = useBodyRectList.size();
    for (int i = 0; i < useBodySize; ++i) {
        if (bodyCount >= bodyNeedCount) {
            // 如果检测到肢体框已经超过需要检测的肢体框的个数，则直接停止循环
            break;
        }
        auto body = result->getBodyResult()->pBodyInfos[useBodyRectList.at(i)];
        bodyCount++;
        VLOGD(TAG, "status_no_face_has_body executed: body old[%d]->new[%d]  faceId[%d]", body->id, bodyCount, -1);
        // 因为判断是否有肢体的逻辑是判断 id > 0,所以body的值从1开始
        body->id = bodyCount;
        // 由于bodyCount是从0开始，所以可以直接使用bodyCount-1作为matchIndex的值
        body->matchIndex = bodyCount - 1;
        bodyRectStrategy->execute(request, nullptr, body, FaceBodyStatus::STATUS_NO_FACE_HAS_BODY);
    }

    // 最后。遍历检测出来人脸框，但是没有检测出来肢体框的逻辑
    auto useFaceSize = useFaceRectList.size();
    for (int i = 0; i < useFaceSize; ++i) {
        if (bodyCount >= bodyNeedCount) {
            // 如果检测到肢体框已经超过需要检测的肢体框的个数，则直接停止循环
            break;
        }
        // 依次遍历所有的肢体框的数据，
        for (int j = 0; j < bodyNeedCount; ++j) {
            auto body = result->getBodyResult()->pBodyInfos[j];
            // 如果当前body的matchIndex的matchIndex为INT_MAX，说明此body之前没有被计算过
            if (body->matchIndex == INT_MAX) {
                auto face = result->getFaceResult()->faceInfos[useFaceRectList.at(i)];
                bodyCount++;
                // 因为判断是否有肢体的逻辑是判断 id > 0,所以body的值从1开始
                VLOGD(TAG, "status_has_face_no_body executed: body old[%d]->new[%d]  faceId[%ld]", body->id, bodyCount,
                      face->id);
                body->id = bodyCount;
                // 由于bodyCount是从0开始，所以可以直接使用bodyCount作为matchIndex的值
                body->matchIndex = bodyCount - 1;
                bodyRectStrategy->execute(request, face, body, FaceBodyStatus::STATUS_HAS_FACE_NO_BODY);
                break;
            }
        }
    }

    // 进行冒泡排序，将人脸按照matchIndex进行重新排序
    for (int i = 0; i < bodyNeedCount - 1; i++) {
        for (int j = 0; j < bodyNeedCount - 1 - i; j++) {
            auto body1 = result->getBodyResult()->pBodyInfos[j];
            auto body2 = result->getBodyResult()->pBodyInfos[j + 1];
            if (body1->matchIndex > body2->matchIndex) {
                VLOGD(TAG, "swap body body1[%d]  and body2[%d]", body1->id, body2->id);
                auto tempBody = body1;
                result->getBodyResult()->pBodyInfos[j] = body2;
                result->getBodyResult()->pBodyInfos[j + 1] = tempBody;
            }
        }
    }

    for (int i = 0; i < bodyNeedCount; ++i) {
        auto body = result->getBodyResult()->pBodyInfos[i];
        VLOGD(TAG, "body_rect[%d] conf=[%f], rect=[%f, %f, %f, %f]", body->id, body->rectConfidence,
              body->headShoulderRectLT.x, body->headShoulderRectLT.y, body->headShoulderRectRB.x,
              body->headShoulderRectRB.y);
    }
}

float BodyRectManager::getMatchIOU(FaceInfo *pFaceInfo, BodyInfo *pBodyInfo) {
    tmpMatchLT.x = std::max(pFaceInfo->rectLT.x, pBodyInfo->headShoulderRectLT.x);
    tmpMatchLT.y = std::max(pFaceInfo->rectLT.y, pBodyInfo->headShoulderRectLT.y);
    tmpMatchRB.x = std::min(pFaceInfo->rectRB.x, pBodyInfo->headShoulderRectRB.x);
    tmpMatchRB.y = std::min(pFaceInfo->rectRB.y, pBodyInfo->headShoulderRectRB.y);

    auto &faceBoxLT = pFaceInfo->rectLT;
    auto &faceBoxRB = pFaceInfo->rectRB;
    // 人脸框和肢体框的默认IOU交叉分数默认为0
    float area_iou = 0.0f;
    float l = std::max(tmpMatchLT.x, faceBoxLT.x);
    float t = std::max(tmpMatchLT.y, faceBoxLT.y);
    float r = std::min(tmpMatchRB.x, faceBoxRB.x);
    float b = std::min(tmpMatchRB.y, faceBoxRB.y);
    if (r <= l || b <= t) {
        area_iou = 0.0f;
    } else {
        float area1 = (tmpMatchRB.x - tmpMatchLT.x) * (tmpMatchRB.y - tmpMatchLT.y);
        float area2 = (faceBoxRB.x - faceBoxLT.x) * (faceBoxRB.y - faceBoxLT.y);
        float area_inter = (r - l) * (b - t);
        area_iou = area_inter / (area1 + area2 - area_inter);
    }
    VLOGD(TAG, "face and body IOU score=%f with face[%ld] and body[%d]", area_iou, pFaceInfo->id, pBodyInfo->id);
    return area_iou;
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("HeadShoulderRectManager", ABILITY_BODY_HEAD_SHOULDER, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<BodyRectManager>());
});

} // namespace vision