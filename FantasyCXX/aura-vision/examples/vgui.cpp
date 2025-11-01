#include "vgui.h"
#include <vector>

#ifdef BUILD_3D_LANDMARK
#include "face_3d_points.h"
#endif

using namespace vision;

namespace vision_demo {
// =========================================== 类别文本标签 ===========================================
// 注意力偏移状态类别
static std::vector<std::string> attentionLabels = {
        "",
        "UnFocused",
        "Uncoordinated",
        "Forward",
        "Left",
        "Right",
        "Up",
        "Down"
};

// 疲劳驾驶状态类别
static std::vector<std::string> fatigueLabels{
        "Normal",
        "Yawn",
        "EyeClose",
        "Yawn+EyeClose"
};

// 眨眼频率状态类别
static std::vector<std::string> eyeBlinkLabels{
        "Unknow",
        "YES",
        "NO"
};

// 危险驾驶状态类别
static std::vector<std::string> dangerLabels{
        "",
        "Normal",
        "Smoke",
        "Silence",
        "Drink",
        "OpenMouth",
        "Eat"
};

// 点头/摇头状态类别
static std::vector<std::string> nodshakeLabels{
        "None",
        "Shake",
        "Nod",
        "Go_on"
};

// 打电话状态类别
static std::vector<std::string> callLabels{
        "Normal",
        "Call"
};

// 表情类别
static std::vector<std::string> emotionLabels{
        "other",
        "happy",
        "surprise",
        "sad",
        "angry"
};

// 活体宠物类别
static std::vector<std::string> categoryLabels{
        "Cat",
        "Dog",
        "Baby"
};

// 眼镜类别
static std::vector<std::string> glassLabels{
        "Sunglasses",
        "NoGlasses",
        "Glasses"
};

// 性别类别
static std::vector<std::string> genderLabels{
        "Male",
        "Female"
};

// 种族类别
static std::vector<std::string> raceLabels{
        "Black",
        "White",
        "Yellow"
};

// 年龄类别
static std::vector<std::string> ageLabels{
        "Baby",
        "Teenager",
        "Youth",
        "Midlife",
        "Senior"
};

// 静态手势类型
static std::vector<std::string> gestureStaticLabels{
        "none",
        "good",
        "ok",
        "dislike",
        "fist",
        "1",
        "2",
        "3",
        "4",
        "5",
        "previous",
        "next",
        "left5",
        "right5",
        "heart",
        "rock"
};

// 动态手势类型
static std::vector<std::string> gestureDynamicLabels{
        "none",
        "wave",
        "flip",
        "pinch",
        "grasp",
        "left_wave",
        "right_wave"
};

// 图像遮挡状态类别
static std::vector<std::string> coverLabels{
        "no",
        "cover"
};

// 跟踪状态类别
static std::vector<std::string> trackLabels{
        "unknown",
        "init",
        "tracking",
        "miss",
};

// 人脸遮挡状态类别
static std::vector<std::string> faceCoverLabels{
        "unknown",
        "leftEye",
        "rightEye",
        "mouth",
        "other",
};

// =========================================== 绘图颜色 ===========================================
//should set path according to detailed setting by self
// 绘制 landmark 的颜色
static cv::Scalar landmarkScalar(255, 0, 0);
// 绘制 landmark 字体的颜色
static cv::Scalar landmarkFontScalar(0, 0, 255);
// 绘制矩形框的颜色
static cv::Scalar rectScalar(0, 255, 255);
// 绘制人眼属性的颜色
static cv::Scalar eyeCircle(0, 255, 0);
// 绘制人眼属性的颜色
static cv::Scalar eyeCircleCmp(0, 0, 255);
// 绘制人脸相关属性文本标签的颜色
static cv::Scalar faceResultColor(0, 255, 0);
// 绘制肢体框的颜色
static cv::Scalar bodyRectScalar(0, 255, 255);

#if defined(BUILD_QNX)
static bool sSaveImageStatus = false;
std::unordered_map<short, bool> GUI::sAbilityMap;
#endif

static void setSaveImageStatus(bool status) {
#if defined(BUILD_QNX)
    sSaveImageStatus = status;
#endif
}

static bool getSaveImageStatus() {
#if defined(BUILD_QNX)
    return sSaveImageStatus;
#else
    return false;
#endif
}

static bool getAbilityCheckSaveImageStatus(short ability) {
#if defined(BUILD_QNX)
    auto it = GUI::sAbilityMap.find(ability);
    if(it != GUI::sAbilityMap.end()) {
        return it->second;
    }
#endif
    return false;
}

void GUI::drawFaceRect(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                       double fontScale, cv::Scalar color, int fontThick, int lineType) {
    float leftUpX = face->rectLT.x;
    float leftUpy = face->rectLT.y;
    int w = face->rectRB.x - leftUpX;
    int h = face->rectRB.y - leftUpy;
    cv::Rect r(leftUpX, leftUpy, w, h);
    cv::rectangle(image, r, color, 3);
    cv::putText(image, "face:" + std::to_string(static_cast<int>(face->id)), cv::Point(leftUpX, leftUpy),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick, lineType);
}

void GUI::drawFaceLandmark(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                           double fontScale, cv::Scalar color, int fontThick, int lineType) {
    cv::Point point;
    cv::Point pointFont;
    for (int j = 0; j < LM_2D_106_COUNT; ++j) {
        point.x = face->landmark2D106[j].x;
        point.y = face->landmark2D106[j].y;
        cv::circle(image, point, 2, color, -1);
        // pointFont.x = point.x;
        // pointFont.y = point.y - 2;
        // cv::putText(image, std::to_string(j), pointFont, cv::FONT_HERSHEY_SIMPLEX, 0.3,
        // landmark_font_scalar, 1,
        //            8);
    }
}

void GUI::drawMouthLandmark(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                           double fontScale, cv::Scalar color, int fontThick, int lineType) {
    cv::Point point;
    cv::Point pointFont;
    for (int j = 0; j < LM_MOUTH_2D_20_COUNT; ++j) {
        point.x = face->mouthLmk20[j].x;
        point.y = face->mouthLmk20[j].y;
        cv::circle(image, point, 2, color, -1);
    }
}

void GUI::drawFaceEyeCentroid(cv::Mat &image, vision::FaceInfo *face, float &leftX, float &leftY, float &rightX,
                              float &rightY, int &xPosition, int &yPosition, int offset, double fontScale,
                              cv::Scalar color, int fontThick, int lineType) {
    // 绘制瞳孔
    std::string text = "not wake";
    if (face->eyeWaking == 1) {
        text = "wake";
    } else {
        text = "not wake";
    }
    cv::putText(image, text, cv::Point(xPosition, xPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick,
                lineType);
    cv::circle(image, cv::Point2f(leftX, leftY), 2, color, 2);
    cv::circle(image, cv::Point2f(rightX, rightY), 2, color, 2);
}

void GUI::drawFaceEyeTracking(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                              double fontScale, cv::Scalar color, int fontThick, int lineType) {
    auto leftEyeCenter = face->eyeTracking[0];
    auto leftEyeGaze = face->eyeTracking[1];
    auto rightEyeCenter = face->eyeTracking[2];
    auto rightEyeGaze = face->eyeTracking[3];
    const int draw_multiplier = 1 << 4;
    const int draw_shiftbits = 0;
#ifdef OPENCV2
    cv::line(image,
             cv::Point(cvRound(leftEyeCenter.x), leftEyeCenter.y),
             cv::Point(cvRound(leftEyeGaze.x), cvRound(leftEyeGaze.y)),
             color,
             fontThick,
             lineType,
             draw_shiftbits);

    cv::line(image,
             cv::Point(cvRound(rightEyeCenter.x), rightEyeCenter.y),
             cv::Point(cvRound(rightEyeGaze.x), cvRound(rightEyeGaze.y)),
             color,
             fontThick,
             lineType,
             draw_shiftbits);
#endif
#ifdef OPENCV4

#endif
}

void GUI::drawFaceNoInteractiveLiving(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition,
                                      int offset, double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "no interactive living: ";
    int status = face->stateNoInteractLiving;
    if (status == F_NO_INTERACT_LIVING_NONE) {
        text = text + "detect no face";
    } else if (status == F_NO_INTERACT_LIVING_LIVING) {
        text = text + "liveness";
    } else {
        text = text + "attack";
    }

    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick,
                lineType);
}

void GUI::drawFaceAttention(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                            double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateAttention);
    if (res >= 0) {
        text += "attention: " + attentionLabels[res];
    }
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
}

void GUI::drawFaceFatigue(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                          double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateFatigue);
    if (res >= 0) {
        text += "fatigue: " + fatigueLabels[res];
    }
    if (face->yawnVState.state >= VSlidingState::START && face->yawnVState.state <= VSlidingState::END) {
        text += ", duration: " + std::to_string(face->yawnVState.continue_time);
    }

    if (face->closeEyeVState.state >= VSlidingState::START && face->closeEyeVState.state <= VSlidingState::END) {
        text += ", duration: " + std::to_string(face->closeEyeVState.continue_time);
    }

    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);

    // eye blink
//    text = "";
//    res = static_cast<int>(face->eyeBlinkState);
//    if (res >= 0) {
//        text += "blink:" + eyeBlinkLabels[res];
//    }
//    text += ",dura:" + std::to_string(cvRound(face->eyeBlinkDuration));
//    text += ",freq:" + std::to_string(cvRound(face->eyeCloseFrequency));
//    yPosition += offset;
//    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
//                color, fontThick);

    // face no moving
//    text = "";
//    res = static_cast<int>(face->stateFaceNoMoving);
//    if (res >= 0) {
//        text += "face no moving: " + eyeBlinkLabels[res];
//    }
//    yPosition += offset;
//    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
//                color, fontThick);
}

void GUI::drawFaceDangerousDriving(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                                   double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateDangerDriveSingle);
    if (res >= 0) {
        text += "danger: " + dangerLabels[res];
    }
    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawFaceCall(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                       double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "call: ";
    if (face->phoneCallVState.state >= VSlidingState::START && face->phoneCallVState.state <= VSlidingState::END) {
        text += "calling, duration: " + std::to_string(face->phoneCallVState.continue_time);
    } else {
        text += "none";
    }
    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
}

void GUI::drawFaceHeadBehavior(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                               double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateHeadBehavior);
    if (res >= 0) {
        text += "head_behave: " + nodshakeLabels[res];
    }
    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
}

void GUI::drawFaceAttribute(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                            double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateGenderSingle);
    if (res >= 0) {
        text += "Gender:" + genderLabels[res];
    }

    res = static_cast<int>(face->stateAgeSingle);
    if (res >= 0 && res < 5) {
        text += ",  Age:" + ageLabels[res];
    }

    res = static_cast<int>(face->stateRaceSingle);
    if (res >= 0) {
        text += ",  Race：" + raceLabels[res];
    }

    res = static_cast<int>(face->stateGlassSingle);
    if (res >= 0) {
        text += ",  Glass:" + glassLabels[res];
    }
    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
}

void
GUI::drawFaceEmotion(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                     double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "emotion: ";
    int res = static_cast<int>(face->stateEmotion);
    if (res >= 0) {
        text += emotionLabels[res];
    } else {
        text += "other";
    }
    yPosition += offset;
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawFaceEyeGaze(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                          double fontScale, cv::Scalar color, int fontThick, int lineType) {
    yPosition += offset;
    std::string text = "head_position [";
    text += std::to_string(face->headLocation.x) + " " +
            std::to_string(face->headLocation.y) +
            " " +
            std::to_string(face->headLocation.z) + "]";
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                faceResultColor, fontThick);
    yPosition += offset;
    text = "head_deflection [" + std::to_string(face->optimizedHeadDeflection.yaw) + " " +
            std::to_string(face->optimizedHeadDeflection.pitch) + " " +
            std::to_string(face->optimizedHeadDeflection.roll) + "]";
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                faceResultColor, fontThick);
    yPosition += offset;

    text = "left eye [";
//    text += std::to_string(face->eyeGazeOriginLeft.x) + " " +
//            std::to_string(face->eyeGazeOriginLeft.y) + " " +
//            std::to_string(face->eyeGazeOriginLeft.z) + "]";
    text += std::to_string(face->eyeGaze3dVectorLeft.x) + " " +
            std::to_string(face->eyeGaze3dVectorLeft.y) + " " +
            std::to_string(face->eyeGaze3dVectorLeft.z) + "]";
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                faceResultColor, fontThick);
    yPosition += offset;

    text = "right eye [";
//    text += std::to_string(face->eyeGazeOriginRight.x) + " " +
//            std::to_string(face->eyeGazeOriginRight.y) + " " +
//            std::to_string(face->eyeGazeOriginRight.z) + "]";
    text += std::to_string(face->eyeGaze3dVectorRight.x) + " " +
            std::to_string(face->eyeGaze3dVectorRight.y) + " " +
            std::to_string(face->eyeGaze3dVectorRight.z) + "]";
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                faceResultColor, fontThick);
    yPosition += offset;
}

void GUI::drawFaceQuality(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                          double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::stringstream text;
    text << "face_cover:";
    if (face->stateFaceCoverSingle) {
        text << " yes [ ";
    } else {
        text << " no [ none";
    }
    if (face->leftEyeCoverSingle) {
        text << faceCoverLabels[1] << " ";
    }
    if (face->rightEyeCoverSingle) {
        text << faceCoverLabels[2] << " ";
    }
    if (face->stateMouthCoverSingle) {
        text << faceCoverLabels[3] << " ";
    }

    text << " ]";
    cv::putText(image, text.str(), cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawSource2CameraCover(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                             double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateCameraCover);
    if (res >= 0) {
        text += "image cover: " + coverLabels[res];
    }
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawSource1CameraCover(cv::Mat &image, vision::FaceInfo *face, int &xPosition, int &yPosition, int offset,
                             double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(face->stateCameraCover);
    if (res >= 0) {
        text += "dms camera cover: " + coverLabels[res];
    }
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawLipMovement(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                          double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "lip movement: ";
    if (face->stateLipMovement == 1) {
        text += "moving";
    } else {
        text += "none";
    }
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawCatDogBaby(cv::Mat &image, vision::LivingInfo *livingInfo, int& xPosition, int& yPosition, int offset,
                         double fontScale, cv::Scalar color, int fontThick, int lineType) {
    std::string text = "";
    int res = static_cast<int>(livingInfo->livingType);
    if (res >= 0) {
        text += "living type: " + categoryLabels[res];
    }
    cv::putText(image, text, cv::Point(xPosition, yPosition), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, fontThick);
    yPosition += offset;
}

void GUI::drawEyeLandmark28(cv::Mat &image, vision::FaceInfo *face, int offset, double fontScale, cv::Scalar color,
                            int fontThick, int lineType) {
    cv::Point point;
    cv::Point pointFont;
    for (int j = 0; j < LM_EYE_2D_28_COUNT; ++j) {
        point.x = face->eye2dLandmark28Left[j].x;
        point.y = face->eye2dLandmark28Left[j].y;
        cv::circle(image, point, 2, cv::Scalar(0, 0, 255), -1);

        point.x = face->eye2dLandmark28Right[j].x;
        point.y = face->eye2dLandmark28Right[j].y;
        cv::circle(image, point, 2, cv::Scalar(0, 0, 255), -1);

//         pointFont.x = point.x;
//         pointFont.y = point.y - 2;
//         cv::putText(image, std::to_string(j), pointFont, cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick,
//                     lineType);
    }
}

void GUI::drawGestureRect(cv::Mat &image, vision::GestureInfo *gesture, int &xPosition, int &yPosition, int offset,
                          double fontScale, cv::Scalar color, int fontThick, int lineType) {
    int w = gesture->rectRB.x - gesture->rectLT.x;
    int h = gesture->rectRB.y - gesture->rectLT.y;
    cv::Rect r(gesture->rectLT.x, gesture->rectLT.y, w, h);
    cv::rectangle(image, r, color, 3);
}

void GUI::drawGestureLandmark(cv::Mat &image, vision::GestureInfo *gesture, int &xPosition, int &yPosition, int offset,
                              double fontScale, cv::Scalar color, int fontThick, int lineType) {
    cv::Point point;
    for (auto &lmk: gesture->landmark21) {
        point.x = lmk.x;
        point.y = lmk.y;
        cv::circle(image, point, 3, landmarkScalar, -1);
    }

    image = drawGestureLandmarkline(image, gesture, rectScalar, 0);
    image = drawGestureLandmarkline(image, gesture, rectScalar, 4);
    image = drawGestureLandmarkline(image, gesture, rectScalar, 8);
    image = drawGestureLandmarkline(image, gesture, rectScalar, 12);
    image = drawGestureLandmarkline(image, gesture, rectScalar, 16);

    std::string GestureType = " ";
    int type = static_cast<int>(gesture->staticType);
    type = std::max(0, type);
    GestureType = gestureStaticLabels[type];
    if (type > 0) {
        cv::putText(image, GestureType, cv::Point(100, 50), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick,
                    lineType);
    }

    int dy_type = static_cast<int>(gesture->dynamicType);
    dy_type = std::max(0, dy_type);
    GestureType = gestureDynamicLabels[dy_type];
    if (type > 0) {
        cv::putText(image, GestureType, cv::Point(200, 50), cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick,
                    lineType);
    }
}

void GUI::drawBodyHeadShouldRect(cv::Mat &image, vision::BodyInfo *body, int &xPosition, int &yPosition, int offset,
                                 double fontScale, cv::Scalar color, int fontThick, int lineType) {
    float leftUpX = body->headShoulderRectLT.x;
    float leftUpy = body->headShoulderRectLT.y;
    int w = body->headShoulderRectRB.x - leftUpX;
    int h = body->headShoulderRectRB.y - leftUpy;
    cv::Rect r(leftUpX, leftUpy, w, h);
    cv::rectangle(image, r, color, 3);
    cv::putText(image, "body:" + std::to_string(static_cast<int>(body->id)), cv::Point(leftUpX, leftUpy),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, color, fontThick, lineType);
}

void GUI::drawFace(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result) {
    auto *faceResult = result->getFaceResult();
    int xPosition = 50;
    int offset = 25;
    double fontScale = 0.6;
    int fontThick = 2;
    int tempXPosition = 0;
    int tempYPosition = 0;
    setSaveImageStatus(false);
    if (!faceResult->noFace()) {
        for (int i = 0; i < faceResult->faceMaxCount; i++) {
            V_CHECK_CONT(faceResult->faceInfos[i]->faceType == FaceDetectType::F_TYPE_UNKNOWN);
            int yPosition = 180;
            xPosition += 350 * i;

            // 绘制人脸框
            if (service->get_switch(ABILITY_FACE_RECT)) {
                drawFaceRect(image, faceResult->faceInfos[i], xPosition, yPosition, offset, 1.5, rectScalar, 4, 8);
            }
            // 绘制人脸landmark
            if (service->get_switch(ABILITY_FACE_LANDMARK)) {
                drawFaceLandmark(image, faceResult->faceInfos[i], xPosition, yPosition, offset, 0.3, landmarkFontScalar,
                                 1, 8);
            }
            // 绘制mouth landmark
            if (service->get_switch(ABILITY_FACE_MOUTH_LANDMARK)) {
                drawMouthLandmark(image, faceResult->faceInfos[i], xPosition, yPosition, offset, 0.3, landmarkFontScalar,
                                 1, 8);
            }
            // 绘制人眼瞳孔
            if (service->get_switch(ABILITY_FACE_EYE_CENTER)) {
                float leftX = 0.0F;
                float leftY = 0.0F;
                float rightX = 0.0F;
                float rightY = 0.0F;
                if (service->get_switch(ABILITY_FACE_EYE_WAKING)) {
                    leftX = faceResult->faceInfos[i]->eyeCentroidLeft.x;
                    leftY = faceResult->faceInfos[i]->eyeCentroidLeft.y;
                    rightX = faceResult->faceInfos[i]->eyeCentroidRight.x;
                    rightY = faceResult->faceInfos[i]->eyeCentroidRight.y;
                } else {
                    leftX = faceResult->faceInfos[i]->eyeTracking[0].x;
                    leftY = faceResult->faceInfos[i]->eyeTracking[0].y;
                    rightX = faceResult->faceInfos[i]->eyeTracking[2].x;
                    rightY = faceResult->faceInfos[i]->eyeTracking[2].y;
                }
                tempXPosition = 100;
                tempYPosition = 100;
//                drawFaceEyeCentroid(image, faceResult->faceInfos[i], leftX, leftY, rightX, rightY, tempXPosition,
//                                    tempYPosition, offset, 2, eyeCircle, 4, 8);
            }
            // 绘制人眼跟踪视线
//            if (service->get_switch(ABILITY_FACE_EYE_GAZE) || service->get_switch(ABILITY_FACE_EYE_TRACKING)) {
//                drawFaceEyeTracking(image, faceResult->faceInfos[i], xPosition, yPosition, offset, 2,
//                                    cv::Scalar(110, 220, 0), 2, 16);
//            }
            // 显示人脸ID
            cv::putText(image, "ID:" + std::to_string(cvRound((double)faceResult->faceInfos[i]->id)), cv::Point(xPosition, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1, rectScalar, 4, 8);
            // 显示制人脸追踪状态
//            if (faceResult->faceInfos[i]->stateFaceTracking > 0) {
//                cv::putText(image, trackLabels[faceResult->faceInfos[i]->stateFaceTracking],
//                            cv::Point(xPosition, 80), cv::FONT_HERSHEY_SIMPLEX, 1, rectScalar, 4, 8);
//            }
            // 显示无感活体检测状态
            if (service->get_switch(ABILITY_FACE_NO_INTERACTIVE_LIVING)) {
                tempYPosition = 110;
                drawFaceNoInteractiveLiving(image, faceResult->faceInfos[i], xPosition, tempYPosition, offset, fontScale,
                                            faceResultColor, fontThick, 8);
            }
            // 显示注意力偏移检测状态
            if (service->get_switch(ABILITY_FACE_ATTENTION)) {
                drawFaceAttention(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                  faceResultColor, fontThick, 8);
            }
            // 显示疲劳驾驶检测状态
            if (service->get_switch(ABILITY_FACE_FATIGUE)) {
                drawFaceFatigue(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                faceResultColor, fontThick, 8);
            }
            // 显示危险驾驶检测状态
            if (service->get_switch(ABILITY_FACE_DANGEROUS_DRIVING)) {
//                drawFaceDangerousDriving(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
//                                         faceResultColor, fontThick, 8);
            }
            // 显示打电话检测状态
            if (service->get_switch(ABILITY_FACE_CALL)) {
                drawFaceCall(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                faceResultColor, fontThick, 8);
            }
            // 显示头部姿态检测状态
            if (service->get_switch(ABILITY_FACE_HEAD_BEHAVIOR)) {
                drawFaceHeadBehavior(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                     faceResultColor, fontThick, 8);
            }
            // 显示人脸属性检测状态
            if (service->get_switch(ABILITY_FACE_ATTRIBUTE)) {
                drawFaceAttribute(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                  faceResultColor, fontThick, 8);
            }
            // 显示人脸表情检测状态
            if (service->get_switch(ABILITY_FACE_EMOTION)) {
                drawFaceEmotion(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                faceResultColor, fontThick, 8);
            }
            // 显示视线检测状态
            if (service->get_switch(ABILITY_FACE_EYE_GAZE)) {
                drawFaceEyeGaze(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                faceResultColor, fontThick, 8);
            }
            // 显示人脸遮挡检测状态
            if (service->get_switch(ABILITY_FACE_QUALITY)) {
                drawFaceQuality(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                faceResultColor, fontThick, 8);
            }
            // 显示OMS图像遮挡检测状态
            if (service->get_switch(ABILITY_CAMERA_COVER)) {
//                drawSource2CameraCover(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
//                                   faceResultColor, fontThick, 8);
            }
            // 显示DMS图像遮挡检测状态
            if (service->get_switch(ABILITY_CAMERA_COVER)) {
                drawSource1CameraCover(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                   faceResultColor, fontThick, 8);
            }

            // 唇动
            if (service->get_switch(ABILITY_FACE_LIP_MOVEMENT)) {
                drawLipMovement(image, faceResult->faceInfos[i], xPosition, yPosition, offset, fontScale,
                                       faceResultColor, fontThick, 8);
            }
#ifdef BUILD_3D_LANDMARK
            drawEyeLandmark28(image, faceResult->faceInfos[i], offset, 0.3, landmarkFontScalar, 1, 8);
#endif
        }
    }
    if (!result->getLivingResult()->livingInfos[0]->hasLiving()) {
        // 猫狗宠物婴儿活体相关检测
        if (service->get_switch(ABILITY_LIVING_DETECTION)) {
            int x = 100, y = 100;
            drawCatDogBaby(image, result->getLivingResult()->livingInfos[0], x, y, offset, fontScale,
                           faceResultColor, fontThick, 8);
        }
    }
}

void GUI::drawGesture(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result) {
    auto *gest_result = result->getGestureResult();
    int xPosition = 50;
    int yPosition = 50;
    int offset = 40;
    double fontScale = 0.6;
    if (!gest_result->noGesture()) {
        GestureInfo *gesture = gest_result->gestureInfos[0];
        if (service->get_switch(ABILITY_GESTURE_RECT)) {
            drawGestureRect(image, gesture, xPosition, yPosition, offset, fontScale, rectScalar, 3, 8);
        }
        if (service->get_switch(ABILITY_GESTURE_LANDMARK)) {
            drawGestureLandmark(image, gesture, xPosition, yPosition, offset, 1, eyeCircle, 4, 8);
        }
    }
}

void GUI::drawBody(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result) {
    auto *bodyResult = result->getBodyResult();
    int xPosition = 50;
    int yPosition = 50;
    int offset = 40;
    double fontScale = 0.6;
    if (bodyResult->hasBody()) {
        for (int i = 0; i < bodyResult->bodyCount(); i++) {
            if (bodyResult->pBodyInfos[i]->id <= 0) {
                continue;
            }
            if (service->get_switch(ABILITY_BODY_HEAD_SHOULDER)) {
                drawBodyHeadShouldRect(image, bodyResult->pBodyInfos[i], xPosition, yPosition, offset, 1.5,
                                       bodyRectScalar, 4, 8);
            }
        }
    }
}

void GUI::drawAllResult(vision::VisionService *service,
                        cv::Mat &image,
                        vision::VisionResult *result) {
    drawFace(service, image, result);
    drawGesture(service, image, result);
    drawBody(service, image, result);
}

cv::Mat &GUI::drawGestureLandmarkline(cv::Mat &image, const vision::GestureInfo *gesture, const cv::Scalar color,
                                      int offset) {
//    line(image, cv::Point(gesture->landmark21[0].x, gesture->landmark21[0].y),
//         cv::Point(gesture->landmark21[1 + offset].x, gesture->landmark21[1 + offset].y), color, 2);
//    line(image, cv::Point(gesture->landmark21[1 + offset].x, gesture->landmark21[1 + offset].y),
//         cv::Point(gesture->landmark21[2 + offset].x, gesture->landmark21[2 + offset].y), color, 2);
//    line(image, cv::Point(gesture->landmark21[2 + offset].x, gesture->landmark21[2 + offset].y),
//         cv::Point(gesture->landmark21[3 + offset].x, gesture->landmark21[3 + offset].y), color, 2);
//    line(image, cv::Point(gesture->landmark21[3 + offset].x, gesture->landmark21[3 + offset].y),
//         cv::Point(gesture->landmark21[4 + offset].x, gesture->landmark21[4 + offset].y), color, 2);
    return image;
}

void GUI::showImg(cv::Mat &image) {
    imshow("image", image);
    cv::waitKey(1);
}

void GUI::saveDrawImg(cv::Mat &image, std::string saveImgPath) {
    cv::imwrite(saveImgPath, image);
}

} // namespace vision_demo