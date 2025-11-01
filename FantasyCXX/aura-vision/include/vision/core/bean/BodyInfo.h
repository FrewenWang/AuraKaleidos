//
// Created by v_zhangtieshuai on 2019-10-08.
//

#ifndef VISION_BODYINFO_H
#define VISION_BODYINFO_H

#include <cstring>
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

/**
 * @brief Body landmark (13 points) definition
 */
enum Body2DLandmark : short {
    //Landmark points in face
    BODY_LM_0_NOSE_TIP1 = 0,
    BODY_LM_1_LEFT_EYE_CENTER2 = 1,
    BODY_LM_2_RIGHT_EYE_CENTER3 = 2,
    BODY_LM_3_LEFT_EAR4 = 3,
    BODY_LM_4_RIGHT_EAR5 = 4,

    //Landmark points in shoulder
    BODY_LM_5_LEFT_SHOULDER6 = 5,
    BODY_LM_6_RIGHT_SHOULDER7 = 6,

    //Landmark points in hand
    BODY_LM_7_LEFT_ELBOW8 = 7,
    BODY_LM_8_RIGHT_ELBOW9 = 8,
    BODY_LM_9_LEFT_WRIST10 = 9,
    BODY_LM_10_LEFT_WRIST11 = 10,
    BODY_LM_11_LEFT_WAIST12 = 11,
    BODY_LM_12_LEFT_WAIST13 = 12,
};

/**
 * @brief has the body been detected
 */
enum BodyStatus : short {
    F_BODY_NONE = 0, // none body
    F_BODY_HAVE = 1  // has body
};

/**
 * @brief The detected Human Position (pedestrain) info
 */
struct BodyInfo {
    static const char LM_2D_7_COUNT = 7;
    int id;                    /// the detected person rect index
    float rectConfidence;         /// head shoulder rect confidence

    //VState noBodyState;           /// has the body been detected
    VPoint headShoulderRectCenter;                      /// the head-shoulder rect of the person
    VPoint headShoulderRectLT;                          /// the left-top head-shoulder rect of the person
    VPoint headShoulderRectRB;                          /// the right-bottom head-shoulder rect of the person
    VPoint bodyLandmark2D12[BODY_LM_2D_12_COUNT];       /// body landmark points (12 points)
    float bodyLandmarkConfidence;                       /// body landmark Confidence (12 points)

    /** 标记当前肢体框是否有匹配上的人脸框,默认为false */
    bool hasMatchedFace = false;
    /** 标记当前人脸匹配之后所在的索引，用于匹配完成之后进行排序,默认为INT_MAX(因为默认无人脸排在最后。索引matchIndex的设置比较大的值). */
    int matchIndex = INT_MAX;

    void copy(const BodyInfo &info);
    /**
     * 判断是否有body。注意：有效肢体框的Id需要从1开始。不能从0开始
     * @return
     */
    bool hasBody() const {
        return id > 0;
    }

    void clearAll();
    /**
     * 清除上一帧的BodyInfo的信息。 @see AbsScheduler.cpp
     */
    void clear();

    BodyInfo() noexcept;

    BodyInfo(const BodyInfo &) noexcept;

    BodyInfo(BodyInfo &&) noexcept;

    BodyInfo &operator=(const BodyInfo &) noexcept;

    BodyInfo &operator=(BodyInfo &&) noexcept;

    ~BodyInfo() = default;

    void toString(std::stringstream &ss) const;
};

} // namespace vision

#endif
