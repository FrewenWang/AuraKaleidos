//
// Created by sunwenming01 on 22-9-26.
//

#pragma once

#include "vision/core/bean/FaceInfo.h"

namespace aura::vision {

/**
 * 嘴部张开幅度工具类
 */
class MouthUtil {

public:
    /** 嘴部张开幅度变化默认阈值 */
    constexpr static float THRESHOLD_LIP_MOVEMENT_DEFAULT = 10.0f;

    /** DMS嘴部张开幅度变化阈值 */
    constexpr static float THRESHOLD_LIP_MOVEMENT_DMS = 10.0f;

    /** OMS嘴部张开幅度变化阈值 */
    constexpr static float THRESHOLD_LIP_MOVEMENT_OMS = 10.0f;

    /**
     * 获取唇部张开距离，上下唇距离取FaceLandmark的100（上嘴唇下）和103（下嘴唇上）之间的距离
     * 此距离为像素绝对距离
     */
    static float getLipDistance(FaceInfo *face);

    /**
     * 获取唇部张开距离，上下唇距离取MouthLandmark的14（上嘴唇下）和17（下嘴唇上）之间的距离
     * 此距离为像素绝对距离
     */
    static float getLipDistanceWithMouthLandmark(FaceInfo *face);

    /**
     * 获取缩放(参照人脸框)到指定尺寸的唇部张开幅度
     */
    static float getLipDistanceRefRect(float lipDistance, float rectHeight);

private:
    /** 人脸框缩放到的高度，用来作为参照计算上下内嘴唇之间的距离 */
    static const short REF_FACE_RECT_HEIGHT = 500;

};

}
