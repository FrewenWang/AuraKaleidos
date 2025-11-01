//
// Created by frewen on 2/3/23.
//

#include "FaceTrackHelper.h"
#include "math_utils.h"
#include "id_util.h"
#include <algorithm>

namespace aura::vision {

const static char *TAG = "FaceTrackHelper";

bool compareFace(const FaceTrackInfo *a, const FaceTrackInfo *b) {
    if (a->matchOrder > 0 && b->matchOrder > 0) {
        return a->matchOrder < b->matchOrder;
    } else if (a->matchOrder <= 0 && b->matchOrder > 0) {
        return false;
    } else if (a->matchOrder > 0 && b->matchOrder <= 0) {
        return true;
    }

    // 需要再讨论丢失帧数多的排在前面还是少的排在前面
    if (a->faceLostCount != b->faceLostCount) {
        return a->faceLostCount > b->faceLostCount;
    }

    if (a->detectType != b->detectType) {
        if (a->detectType == FaceDetectType::F_TYPE_DETECT) {
            return true;
        } else {
            return false;
        }
    }

    if (a->rectConfidence != b->rectConfidence) {
        return a->rectConfidence > b->rectConfidence;
    }

    // 当丢失超过8帧后仅保留人脸id，并按id从大到小排序
    return a->faceIndex > b->faceIndex;
}

bool compareFaceWithDistance(const FaceTrackInfo *a, const FaceTrackInfo *b) {
    if (a->eulerDistance > 0 && b->eulerDistance > 0) {
        return a->eulerDistance < b->eulerDistance;
    } else if (a->eulerDistance <= 0 && b->eulerDistance > 0) {
        return false;
    } else if (a->eulerDistance > 0 && b->eulerDistance <= 0) {
        return true;
    }

    if (a->matchOrder > 0 && b->matchOrder > 0) {
        return a->matchOrder < b->matchOrder;
    } else if (a->matchOrder <= 0 && b->matchOrder > 0) {
        return false;
    } else if (a->matchOrder > 0 && b->matchOrder <= 0) {
        return true;
    }

    // 需要再讨论丢失帧数多的排在前面还是少的排在前面
    if (a->faceLostCount != b->faceLostCount) {
        return a->faceLostCount > b->faceLostCount;
    }

    if (a->detectType != b->detectType) {
        if (a->detectType == FaceDetectType::F_TYPE_DETECT) {
            return true;
        } else {
            return false;
        }
    }

    if (a->rectConfidence != b->rectConfidence) {
        return a->rectConfidence > b->rectConfidence;
    }

    // 当丢失超过8帧后仅保留人脸id，并按id从大到小排序
    return a->faceIndex > b->faceIndex;
}

/**
 * 对模型检出人脸按距主驾距离、置信度排序
 * @param a 临时人脸信息数据
 * @param b 临时人脸信息数据
 * @return
 */
bool compareCurFaceWithDistance(const CurFaceInfo *a, const CurFaceInfo *b) {
    if (a->eulerDistance > 0 && b->eulerDistance > 0) {
        return a->eulerDistance < b->eulerDistance;
    } else if (a->eulerDistance <= 0 && b->eulerDistance > 0) {
        return false;
    } else if (a->eulerDistance > 0 && b->eulerDistance <= 0) {
        return true;
    }

    return a->faceInfo->rectConfidence > b->faceInfo->rectConfidence;
}

/**
 * 此处查看 trackFacesList中跟踪的具体信息以方便排查问题，后续稳定关闭
 * @param trackFacesList
 */
void printTrackFacesList(const std::vector<FaceTrackInfo *> *trackFacesList, std::stringstream &stream) {
    stream.clear();
    stream.str("");
    for (auto trackFace: *trackFacesList) {
        if (trackFace->detectType == FaceDetectType::F_TYPE_UNKNOWN) {
            break ;
        }
        stream << "face[" << std::to_string(trackFace->faceIndex) << "], "
               << "type=[" << std::to_string(trackFace->detectType) << "], "
               << "lost=[" << std::to_string(trackFace->faceLostCount) << "], "
               << "order=[" << std::to_string(trackFace->matchOrder) << "], "
               << "conf=[" << std::to_string(trackFace->rectConfidence) << "], "
               << "distance=[" << std::to_string(trackFace->eulerDistance) << "], "
               << "rect=[" << std::to_string(trackFace->rectLT.x) << "," << std::to_string(trackFace->rectLT.y)
               << "," << std::to_string(trackFace->rectRB.x) << "," << std::to_string(trackFace->rectRB.y)
               << "]";
        VLOGD(TAG, "%s", stream.str().c_str());
    }
}

void CurFaceInfo::clear() {
    eulerDistance = 0.f;
    faceInfo = nullptr;
}

void FaceTrackInfo::clear() {
    faceIndex = 0;
    eulerDistance = 0.f;
    detectType = FaceDetectType::F_TYPE_UNKNOWN;
    faceLostCount = FaceTrackHelper::DEFAULT_FACE_LOST_COUNT;
    rectConfidence = 0.f;
    matchOrder = 0;
    faceRect.clear();
    rectLT.clear();
    rectRB.clear();
}

void FaceTrackInfo::recycle() {
    clear();
    ObjectPool::recycle(this);
}

void FaceTrackHelper::init(RtConfig *cfg) {
    mRtConfig = cfg;
    for (int i = 0; i < mRtConfig->faceMaxCount; ++i) {
        curOrderedFaces.push_back(new CurFaceInfo());
        trackFacesList.push_back(new FaceTrackInfo());
    }
}

void FaceTrackHelper::reset() {
    faceCountCurFrame = 0;
    matchedFaceIndices.clear();
    noMatchedFaceIndices.clear();
    // 清空之前CurFaceInfo数据，存储当前帧FaceInfo地址信息
    for (int i = 0; i < V_TO_INT(curOrderedFaces.size()); ++i) {
        curOrderedFaces[i]->clear();
    }
    // 将所有有人脸的 FaceTrackInfo 的类型设置为跟踪TRACK；人脸连续丢失超过最大帧数则清除其信息
    preFaceCount = 0; // 每一帧都需要重新计算上一帧的人脸数量
    auto trackFaceLen = static_cast<int>(trackFacesList.size());
    for (auto i = 0; i < trackFaceLen; ++i) {
        auto trackFace = trackFacesList[i];
        // 每次将trackFacesList的每个FaceTrackInfo的过往匹配次序清除
        trackFace->matchOrder = 0;
        if (trackFace->detectType != FaceDetectType::F_TYPE_UNKNOWN) {
            trackFace->detectType = FaceDetectType::F_TYPE_TRACK;
            if (++trackFace->faceLostCount >= MAX_FACE_LOST_COUNT) {
                trackFace->detectType = F_TYPE_UNKNOWN;
                trackFace->faceLostCount = FaceTrackHelper::DEFAULT_FACE_LOST_COUNT;
                trackFace->rectConfidence = 0;
                trackFace->eulerDistance = 0.f;
                trackFace->faceRect.clear();
                trackFace->rectLT.clear();
                trackFace->rectRB.clear();
            } else {
                preFaceCount++;
            }
        }
    }
}

float FaceTrackHelper::getDriverRoiDistance(const FaceInfo *faceInfo) {
    auto centerPosX = (faceInfo->rectLT.x + faceInfo->rectRB.x) / 2.0f;
    auto centerPosY = (faceInfo->rectLT.y + faceInfo->rectRB.y) / 2.0f;
    float eulerDistance = powf((powf(centerPosX - mRtConfig->driverRoiPositionX, 2)
                                + powf(centerPosY - mRtConfig->driverRoiPositionY, 2)), 0.5);
    VLOGD(TAG, "roiX[%f] roiY[%f] id[%d] centerPosX[%f] centerPosY[%f] eulerDistance[%f]",
          mRtConfig->driverRoiPositionX, mRtConfig->driverRoiPositionY,
          faceInfo->id, centerPosX, centerPosY, eulerDistance);
    return eulerDistance;
}

void FaceTrackHelper::faceCopyRoiFaceInfo(VisionResult *result) {
    int faceNeedCount = static_cast<int>(mRtConfig->faceNeedDetectCount);
    int faceMaxCount = static_cast<int>(mRtConfig->faceMaxCount);
    auto infos = result->getFaceResult()->faceInfos;
    // 将FaceInfo地址依次填充到curOrderedFaces集中，并计算对应的ROI距离
    for (int i = 0; i < faceMaxCount; ++i) {
        auto iter = curOrderedFaces.at(i);
        iter->faceInfo = infos[i];
        if (iter->faceInfo->faceType == FaceDetectType::F_TYPE_UNKNOWN) {
            iter->eulerDistance = 0;
        } else {
            iter->eulerDistance = getDriverRoiDistance(infos[i]);
        }
    }
    std::sort(curOrderedFaces.begin(), curOrderedFaces.end(), compareCurFaceWithDistance);
    // 将排序后的地址更新到result中
    for (int i = 0; i < faceMaxCount; ++i) {
        infos[i] = curOrderedFaces.at(i)->faceInfo;
        // Landmark中仅操作前faceNeedCount个人脸，因此将多余人脸置为无效人脸
        if (i >= faceNeedCount && infos[i]->faceType != FaceDetectType::F_TYPE_UNKNOWN) {
            infos[i]->faceType = FaceDetectType::F_TYPE_UNKNOWN;
        }
    }
}

bool FaceTrackHelper::faceCopyFaceInfo(const VisionResult *result) {
    int faceNeedCount = static_cast<int>(mRtConfig->faceNeedDetectCount);
    int faceMaxCount = static_cast<int>(mRtConfig->faceMaxCount);
    auto infos = result->getFaceResult()->faceInfos;
    // 将FaceInfos前faceNeedCount个地址依次填充到curOrderedFaces集合中，并多余人脸置为无效人脸
    for (int i = 0; i < faceMaxCount; ++i) {
        curOrderedFaces.at(i)->faceInfo = infos[i];
        if (i >= faceNeedCount && infos[i]->faceType != FaceDetectType::F_TYPE_UNKNOWN) {
            infos[i]->faceType = FaceDetectType::F_TYPE_UNKNOWN;
        }
    }
    return true;
}

bool FaceTrackHelper::faceTrackAndMatch(VisionResult *result) {
    // 获取跟踪列表中的人脸的数量
    trackFaceCount = static_cast<int>(trackFacesList.size());
    faceCountCurFrame = static_cast<int>(curOrderedFaces.size());
    VLOGD(TAG, "trackFaceCount[%d] faceCountCurFrame[%d]", trackFaceCount, faceCountCurFrame);

    // 将模型检测的人脸一一与跟踪列表中人脸进行匹配
    faceMatch();
    // 如果模型检出的人脸有没匹配到的，同时跟踪列表中还有空位，则将没匹配到的人脸尽可能加入到跟踪列表中
    faceAdd();

    // 如果业务侧，开启了主驾ROI区域过滤。 获取到所有临时人脸数据之后，按照人脸距离的ROI position由近到远排序
    if (V_F_TO_BOOL(mRtConfig->useDriverRoiPositionFilter)) {
        std::sort(trackFacesList.begin(), trackFacesList.end(), compareFaceWithDistance);
    } else {
        std::sort(trackFacesList.begin(), trackFacesList.end(), compareFace);
    }
    printTrackFacesList(&trackFacesList, stream);

    // 遍历所有跟踪的人脸。然后将跟踪的人脸赋值到人脸列表
    faceCopyTrackList(result->getFaceResult()->faceInfos);
    return true;
}

void FaceTrackHelper::faceMatch() {
    // 匹配成功的次序
    order = 0;
    // 遍历所有当前帧检测到的人脸和跟踪的人脸列表中进行一一匹配。
    for (int i = 0; i < faceCountCurFrame; ++i) {
        auto detectFace = curOrderedFaces[i]->faceInfo;
        if (detectFace == nullptr || detectFace->faceType == FaceDetectType::F_TYPE_UNKNOWN) {
            continue ;
        }
        float maxIouScore = 0.f;
        int cacheIndex = -1;
        for (int j = 0; j < trackFaceCount; ++j) {
            auto trackFace = trackFacesList[j];
            // 如果跟踪的人脸已经被检测到人脸匹配完成，或者是默认UNKNOWN则不需要再被匹配
            V_CHECK_CONT(trackFace->detectType != FaceDetectType::F_TYPE_TRACK);
            auto score = MathUtils::base_iou(detectFace->faceRect, trackFace->faceRect);
            if (score > maxIouScore) {
                maxIouScore = score;
                cacheIndex = j;
            }
        }
        // 如果当前检测到人脸匹配的Index>=0.则证明当前检测到的人脸和跟踪人脸匹配成功
        if (cacheIndex >= 0) {
            auto trackFace = trackFacesList[cacheIndex];
            trackFace->matchOrder = ++order;
            trackFace->detectType = FaceDetectType::F_TYPE_DETECT;
            trackFace->rectConfidence = detectFace->rectConfidence;
            trackFace->eulerDistance = curOrderedFaces[i]->eulerDistance;
            trackFace->faceRect.copy(detectFace->faceRect);
            trackFace->rectLT.copy(detectFace->rectLT);
            trackFace->rectRB.copy(detectFace->rectRB);
            matchedFaceIndices.push_back(i);
            VLOGD(TAG, "detectFace[%d] and trackFace[%d] matched IOU score[%f]", detectFace->id,
                  trackFace->faceIndex, maxIouScore);
        } else {
            noMatchedFaceIndices.push_back(i);
            VLOGD(TAG, "detectFace[%d] no matched", detectFace->id);
        }
    }
}

void FaceTrackHelper::faceAdd() {
    auto noMatchedCount = static_cast<int>(noMatchedFaceIndices.size());
    VLOGD(TAG, "noMatchedCount[%d] preFaceCount[%d] trackFaceCount[%d]", noMatchedCount, preFaceCount, trackFaceCount);
    // 理论上添加到列表的最尾部是比较合理的，但是目前不想频繁进行人脸的移除和添加。
    if (noMatchedCount > 0 && preFaceCount < trackFaceCount) {
        for (auto iter = noMatchedFaceIndices.begin(); iter != noMatchedFaceIndices.end(); iter++) {
            auto detectFace = curOrderedFaces[*iter]->faceInfo;
            for (int k = 0; k < trackFaceCount; ++k) {
                auto trackFace = trackFacesList[k];
                if (trackFace->detectType == FaceDetectType::F_TYPE_UNKNOWN) {
                    trackFace->detectType = FaceDetectType::F_TYPE_DETECT;
                    trackFace->faceIndex = FaceIdUtil::instance()->produce();
                    trackFace->rectConfidence = detectFace->rectConfidence;
                    trackFace->eulerDistance = curOrderedFaces[*iter]->eulerDistance;
                    trackFace->faceLostCount = FaceTrackHelper::DEFAULT_FACE_LOST_COUNT;
                    trackFace->faceRect.copy(detectFace->faceRect);
                    trackFace->rectLT.copy(detectFace->rectLT);
                    trackFace->rectRB.copy(detectFace->rectRB);
                    VLOGD(TAG, "trackFaceList add new detectFace=[%d], confidence=[%f]", detectFace->id,
                          trackFace->rectConfidence);
                    break;
                }
            }
        }
    }
}

bool FaceTrackHelper::faceCopyTrackList(FaceInfo **pInfo) {
    auto trackFaceCount = static_cast<int>(trackFacesList.size());
    int needDetectCount = static_cast<int>(mRtConfig->faceNeedDetectCount);
    // 排序过后所有id不为0的都排在前面
    for (int m = 0; m < trackFaceCount; ++m) {

        // 跟踪列表中无效人脸的LostCount置为-1，区分landmark矫正重置的F_TYPE_UNKNOWN人脸（LostCount > -1）
        auto trackFace = trackFacesList[m];
        if (trackFace->detectType == FaceDetectType::F_TYPE_UNKNOWN) {
            trackFace->faceLostCount = FaceTrackHelper::DEFAULT_FACE_LOST_COUNT;
        }

        // 将needDetectCount数量之外人脸置为无效人脸
        auto face = pInfo[m];
        if (m >= needDetectCount) {
            // 清除向外暴露的无效FaceInfo的所有信息，避免其变为有效时，携带了过去的错误信息
            face->clearAll();
            continue;
        }
        // 将跟踪信息填充到FaceInfos的对应人脸
        face->id = trackFace->faceIndex;
        face->rectConfidence = trackFace->rectConfidence;
        face->faceRect.copy(trackFace->faceRect);
        face->rectLT.copy(trackFace->rectLT);
        face->rectRB.copy(trackFace->rectRB);
        face->faceType = trackFace->detectType;
        VLOGD(TAG, "face_rect[%ld] confidence=[%f],detectType=[%d] rect=[%f,%f,%f,%f]", face->id,
              trackFace->rectConfidence, face->faceType, face->rectLT.x, face->rectLT.y, face->rectRB.x,
              face->rectRB.y);
    }
    return true;
}


FaceTrackHelper::~FaceTrackHelper() {
    preFaceCount = 0;
    faceCountCurFrame = 0;
    trackFaceCount = 0;
    order = 0;
    matchedFaceIndices.clear();
    noMatchedFaceIndices.clear();
    stream.clear();
    // 回收curOrderedFaces所有的数据
    for (auto it = curOrderedFaces.begin(); it != curOrderedFaces.end(); it++) {
        if (*it != nullptr) {
            delete *it;
            *it = nullptr;
        }
    }
    // 回收trackFacesList所有的数据
    for (auto it = trackFacesList.begin(); it != trackFacesList.end(); it++) {
        if (*it != nullptr) {
            delete *it;
            *it = nullptr;
        }
    }
}

} // namespace aura::vision
