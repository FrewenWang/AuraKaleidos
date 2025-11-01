
#ifndef VISION_FACE_EYE_TRACKING_MANAGER_H
#define VISION_FACE_EYE_TRACKING_MANAGER_H

#include "opencv2/opencv.hpp"

#include "vision/manager/AbsVisionManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"

namespace aura::vision {
/**
 * @brief 人脸眼球追踪管理器
 * */
class FaceEyeTrackingManager : public AbsVisionManager {
public:
    FaceEyeTrackingManager();

    ~FaceEyeTrackingManager() override = default;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:

    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    void eye_tracking_detector(int width, int height, unsigned char *frame, FaceInfo *fi);

    std::vector<std::pair<cv::Point2f, cv::Point2f> > return_box(FaceInfo *face_info,
                                                                 cv::Mat &vec_rot, cv::Mat &vec_trans);

    void eye_site_3d(FaceInfo *face_info, VPoint eye_site2d, VPoint3 &eye_site3d,
                     VPoint &eye_site_front, cv::Mat &vec_rot, cv::Mat &vec_trans, bool is_right_eye);

    void reset_two_points(VPoint &p1, VPoint &p2);

    void drawline(VPoint p1, VPoint p2, VEyeInfo roi_eye, int value, char *mask);

    bool set_pixel(int x, int y, VEyeInfo roi_eye, int value, char *mask);

    void project(cv::Mat &rot_box_proj, cv::Mat rot_box);

private:
    float _fx;
    float _fy;
    float _cx;
    float _cy;
    int _width;
    int _height;
    VPoint3 _right_eye_site3d;
    VPoint3 _left_eye_site3d;
    VPoint _right_eye_front;
    VPoint _left_eye_front;
    VPoint _right_eye_site;
    VPoint _left_eye_site;
    cv::Mat _vec_trans;
    cv::Mat _vec_rot;

    //    const float _k_ccd_focal_length_pixel_x =
    //        _cfg->get_config(CAMERA_FOCAL_LENGTH)/ (_cfg->get_config(CAMERA_CCD_WIDTH) / _cfg->get_config(FRAME_WIDTH));
    //    const float _k_ccd_focal_length_pixel_y =
    //        _cfg->get_config(CAMERA_FOCAL_LENGTH)/ (_cfg->get_config(CAMERA_CCD_HEIGHT) / _cfg->get_config(FRAME_HEIGHT));
};
};

#endif //VISION_FACE_EYE_TRACKING_MANAGER_H
