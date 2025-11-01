//
// Created by sunwenming01 on 22-9-26.
//

#include "vision/util/MouthUtil.h"

namespace aura::vision {

static const char *TAG = "MouthUtil";

float MouthUtil::getLipDistance(FaceInfo *face) {
    // 上下唇距离取100（上嘴唇下）和103（下嘴唇上）之间的距离
    auto point100 = face->landmark2D106[FLM_99_MOUTH_TOP_LIP_BOTTOM_TOP];
    auto point103 = face->landmark2D106[FLM_102_MOUTH_LOWER_LIP_TOP_BOTTOM];
    auto res = (float) sqrt(pow(point103.x - point100.x, 2) + pow(point103.y - point100.y, 2));
    VLOGD(TAG, "2d landmark point100(%f, %f) point103(%f, %f) res(%f)",
          point100.x, point100.y, point103.x, point103.y, res);
    return res;
}

float MouthUtil::getLipDistanceWithMouthLandmark(FaceInfo *face) {
    // 使用单独的嘴部关键点模型结果，(索引下标)FaceLandmark的99 - MouthLandmark的13, FaceLandmark的102 - MouthLandmark的16
    auto point100 = face->mouthLmk20[MouthLandmark::MLM_13_MOUTH_TOP_LIP_BOTTOM_TOP];
    auto point103 = face->mouthLmk20[MouthLandmark::MLM_16_MOUTH_LOWER_LIP_TOP_BOTTOM];
    auto res = (float) sqrt(pow(point103.x - point100.x, 2) + pow(point103.y - point100.y, 2));
    VLOGD(TAG, "mouth landmark point100(%f, %f) point103(%f, %f) res(%f)",
          point100.x, point100.y, point103.x, point103.y, res);
    return res;
}

float MouthUtil::getLipDistanceRefRect(float lipDistance, float rectHeight) {
    auto res = (float) MouthUtil::REF_FACE_RECT_HEIGHT * lipDistance / rectHeight;
    VLOGD(TAG, "lipDistance(%f) rectHeight(%f) res(%f)", lipDistance, rectHeight, res);
    return res;
}

}
