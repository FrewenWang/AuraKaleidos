#pragma once

#include "opencv2/opencv.hpp"
#include "../include/vision/VisionAbility.h"
#include "vision/core/bean/GestureInfo.h"
#include <unordered_map>

namespace vision_demo {

class GUI {
public:
    // =========================================== 单独绘制人脸检测相关结果 ===========================================
    /**
     * 绘制人脸框
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceRect(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                             double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制人脸 Landmark
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceLandmark(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                 double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制Mouth Landmark
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawMouthLandmark(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                 double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制瞳孔中心点
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceEyeCentroid(cv::Mat &image, vision::FaceInfo *face, float& leftX, float& leftY, float& rightX,
                                    float& rightY, int& xPosition, int& yPosition, int offset, double fontScale,
                                    cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制视线跟踪时瞳孔中心点和视线
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceEyeTracking(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                    double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制无感活体检测结果
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceNoInteractiveLiving(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                            double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制注意力状态
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceAttention(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                  double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制疲劳驾驶状态
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceFatigue(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制危险驾驶状态
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceDangerousDriving(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                         double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制打电话状态
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceCall(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                             double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制头部姿态结果
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceHeadBehavior(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                     double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制人脸属性
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceAttribute(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                  double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制人脸表情结果
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceEmotion(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制视线检测结果
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceEyeGaze(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制人脸质量
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawFaceQuality(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制人脸质量
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawSource1CameraCover(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制唇动结果
     */
    static void drawLipMovement(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                       double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制图像遮挡检测结果
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawSource2CameraCover(cv::Mat &image, vision::FaceInfo *face, int& xPosition, int& yPosition, int offset,
                                   double fontScale, cv::Scalar color, int fontThick, int lineType);

    static void drawCatDogBaby(cv::Mat &image, vision::LivingInfo *livingInfo, int& xPosition, int& yPosition, int offset,
                               double fontScale, cv::Scalar color, int fontThick, int lineType);

    /**
     * 绘制眼睛 landmark
     * @param image         待绘制图像
     * @param face          人脸检测结果
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawEyeLandmark28(cv::Mat &image, vision::FaceInfo *face, int offset, double fontScale,
                                  cv::Scalar color, int fontThick, int lineType);

    // =========================================== 单独绘制手势检测相关结果 ===========================================
    /**
     * 绘制手势框
     * @param image         待绘制图像
     * @param gesture       手势检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawGestureRect(cv::Mat &image, vision::GestureInfo *gesture, int& xPosition, int& yPosition, int offset,
                                double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 绘制手势 landmark
     * @param image         待绘制图像
     * @param gesture       手势检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawGestureLandmark(cv::Mat &image, vision::GestureInfo *gesture, int& xPosition, int& yPosition, int offset,
                                    double fontScale, cv::Scalar color, int fontThick, int lineType);
    /**
     * 将手势关键点用线条连接起来
     * @param image     待绘制图像
     * @param gesture   手势检测结果
     * @param color     绘制使用颜色
     * @param offset    下标偏移量
     * @return
     */
    static cv::Mat &drawGestureLandmarkline(cv::Mat &image, const vision::GestureInfo *gesture,
                                            const cv::Scalar color, int offset);

    // =========================================== 单独绘制肢体检测相关结果 ===========================================
    /**
     * 绘制头肩框
     * @param image         待绘制图像
     * @param body          肢体检测结果
     * @param xPosition     文本标签在图像上X轴方向位置
     * @param yPosition     文本标签在图像上Y轴方向位置
     * @param offset        位置偏移量
     * @param fontScale     绘制使用字体大小
     * @param color         绘制使用颜色
     * @param fontThick     绘制使用字体粗细
     * @param lineType      线条类型
     */
    static void drawBodyHeadShouldRect(cv::Mat &image, vision::BodyInfo *body, int& xPosition, int& yPosition,
                                       int offset, double fontScale, cv::Scalar color, int fontThick, int lineType);

    // =========================================== 绘制各result相关结果 ===========================================
    /**
     * 绘制 faceResult 结果
     * @param service   VisionService
     * @param image     待绘制图像
     * @param result    faceResult
     * @param isShow    是否显示绘制图像
     */
    static void drawFace(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result);
    /**
     * 绘制 gestureResult 结果
     * @param service   VisionService
     * @param image     待绘制图像
     * @param result    gestureResult
     * @param isShow    是否显示绘制图像
     */
    static void drawGesture(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result);
    /**
     * 绘制 bodyResult 结果
     * @param service   VisionService
     * @param image     待绘制图像
     * @param result    bodyResult
     * @param isShow    是否显示绘制图像
     */
    static void drawBody(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result);

    // =========================================== 绘制所有result相关结果 ===========================================
    /**
     * 绘制所有 Result 结果
     * @param service   VisionService
     * @param image     待绘制图像
     * @param result    VisionResult
     * @param isShow    是否显示绘制图像
     */
    static void drawAllResult(vision::VisionService *service, cv::Mat &image, vision::VisionResult *result);

    /**
     * 显示图像
     * @param image 绘制后的图像
     */
    static void showImg(cv::Mat &image);

    /**
     * 保存绘制后的图像到指定路径
     * @param image         绘制后的图像
     * @param saveImgPath   保存图像的路径
     */
    static void saveDrawImg(cv::Mat &image, std::string saveImgPath);
#if defined(BUILD_QNX)
public:
    static std::unordered_map<short, bool> sAbilityMap;
#endif
};

} // namespace vision_demo