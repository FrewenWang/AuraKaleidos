
#ifdef BUILD_EXPERIMENTAL

#include "FaceEyeTrackingManager.h"

#include <vector>

#include "vision/config/runtime_config/RtConfig.h"
//#include "deprecated_detector/iov/ncnn/iov_ncnn_face_eye_gaze_detector.h"
#include "util/FaceEyeHelper.h"
#include "util/lmk_converter.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

const static std::string TAG = "FaceEyeTrackingManager";

FaceEyeTrackingManager::FaceEyeTrackingManager() {

}

void FaceEyeTrackingManager::init(RtConfig *cfg) {
    if (cfg) {
        AbsVisionManager::init(cfg);
        mRtConfig = cfg;
        _fx = mRtConfig->cameraFocalLength / (mRtConfig->cameraCcdWidth / mRtConfig->frameWidth);
        _fy = mRtConfig->cameraFocalLength / (mRtConfig->cameraCcdHeight / mRtConfig->frameHeight);
        _cx = mRtConfig->frameWidth / 2.0;
        _cy = mRtConfig->frameHeight / 2.0;
        _width = 0;
        _height = 0;
        _vec_trans.create(3, 1, CV_64FC1);
        _vec_rot.create(3, 1, CV_64FC1);
    }
}

void FaceEyeTrackingManager::deinit() {
    AbsVisionManager::deinit();
}

bool FaceEyeTrackingManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_EYE_TRACKING);
}

void FaceEyeTrackingManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_EYE_TRACKING);

    auto* face_result = result->getFaceResult();
    FaceInfo *fi = face_result->faceInfos[0];
    // 判断当前人脸是否是模型检出
    if (fi->faceType != FaceDetectType::F_TYPE_DETECT) {
        return;
    }
    if (!result->isAbilityExec(ABILITY_FACE_LANDMARK_68P)) {
        LmkConverter::get_68_point(fi);
        result->setAbilityExec(ABILITY_FACE_LANDMARK_68P);
    }

//        if (!result->is_ability_exec(ABILITY_FACE_2DTO3D)) {
//            PERF_TICK(result, "face_2dto3d")
//            IovNcnnFace2Dto3DDetector::instance()->detect(request->_width, request->_height, request->_frame,
//                    result->_face_count, result->_face_infos, result->get_perf_util());
//
//            PERF_TOCK(result, "face_2dto3d")
//            result->set_ability_exec(ABILITY_FACE_2DTO3D);
//        }
//
//        // 眼球追踪策略
//        PERF_TICK(result, "face_eye")
//        //输出眼球2D点
//        FaceEyeHelper::instance()->detect(request, result);
//        _left_eye_site = result->_face_infos[0]->_eye_centroid_left;
//        _right_eye_site = result->_face_infos[0]->_eye_centroid_right;
//
//        eye_tracking_detector(request->_width, request->_height, request->_frame, fi);
//        PERF_TOCK(result, "face_eye")

//    PERF_TICK(result->get_perf_util(), "eye_tracking");
//    IovNcnnFaceEyeGazeDetector::instance()->detect(request->_width, request->_height, request->_frame,
//                                                   face_result->_face_count, face_result->_face_infos, result->get_perf_util());
//    PERF_TOCK(result->get_perf_util(), "eye_tracking");
}

void FaceEyeTrackingManager::eye_tracking_detector(int width, int height, unsigned char *frame, FaceInfo *fi) {
    return_box(fi, _vec_rot, _vec_trans);
    //输出眼球3D
    eye_site_3d(fi, _right_eye_site, _right_eye_site3d, _right_eye_front, _vec_rot, _vec_trans, true);
    eye_site_3d(fi, _left_eye_site, _left_eye_site3d, _left_eye_front, _vec_rot, _vec_trans, false);

    fi->eyeTracking[0].x = _right_eye_site.x;
    fi->eyeTracking[0].y = _right_eye_site.y;

    fi->eyeTracking[1].x = _left_eye_site.x;//界面显示这个点的坐标
    fi->eyeTracking[1].y = _left_eye_site.y;

    fi->eyeTracking[2].x = _right_eye_front.x;//界面显示这个终点
    fi->eyeTracking[2].y = _right_eye_front.y;

    fi->eyeTracking[3].x = _left_eye_front.x;//界面显示这个终点
    fi->eyeTracking[3].y = _left_eye_front.y;
}

cv::Matx33f euler2rotation_matrix(const cv::Vec3f &euler_angles) {
    cv::Matx33f rotation_matrix;

    float s1 = sin(euler_angles[0]);
    float s2 = sin(euler_angles[1]);
    float s3 = sin(euler_angles[2]);

    float c1 = cos(euler_angles[0]);
    float c2 = cos(euler_angles[1]);
    float c3 = cos(euler_angles[2]);

    rotation_matrix(0, 0) = c2 * c3;
    rotation_matrix(0, 1) = -c2 * s3;
    rotation_matrix(0, 2) = s2;
    rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
    rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
    rotation_matrix(1, 2) = -c2 * s1;
    rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
    rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
    rotation_matrix(2, 2) = c1 * c2;

    return rotation_matrix;
}

std::vector<std::pair<cv::Point2f, cv::Point2f> >
FaceEyeTrackingManager::return_box(FaceInfo *face_info, cv::Mat &vec_rot, cv::Mat &vec_trans) {

    cv::Mat landmarks68_2d(LM_2D_68_COUNT, 2, CV_64FC1);
    cv::Mat landmarks68_3d(LM_3D_68_COUNT, 3, CV_64FC1);

    for (int i = 0; i < LM_2D_68_COUNT; ++i) {
        landmarks68_3d.at<double>(i, 0) = face_info->landmark3D68[i].x;
        landmarks68_3d.at<double>(i, 1) = face_info->landmark3D68[i].y;
        landmarks68_3d.at<double>(i, 2) = face_info->landmark3D68[i].z;
        landmarks68_2d.at<double>(i, 0) = face_info->landmark2D68[i].x;
        landmarks68_2d.at<double>(i, 1) = face_info->landmark2D68[i].y;
    }

    double camd[9] = {_fx, 0, mRtConfig->frameWidth / 2.0,
                      0, _fy, mRtConfig->frameHeight / 2.0,
                      0, 0, 1};
    cv::Mat camera_matrix = cv::Mat(3, 3, CV_64FC1, camd);

    cv::solvePnP(landmarks68_3d, landmarks68_2d, camera_matrix, cv::Mat_<double>(), vec_rot, vec_trans, false);
    landmarks68_2d.release();
    landmarks68_3d.release();
    camera_matrix.release();

    // The size of the head is roughly 200mm x 200mm x 200mm
    float box_verts[] = {-1, 1, -1,
                         1, 1, -1,
                         1, 1, 1,
                         -1, 1, 1,
                         1, -1, 1,
                         1, -1, -1,
                         -1, -1, -1,
                         -1, -1, 1};

    std::vector<std::pair<int, int> > edges;
    edges.push_back(std::pair<int, int>(0, 1));
    edges.push_back(std::pair<int, int>(1, 2));
    edges.push_back(std::pair<int, int>(2, 3));
    edges.push_back(std::pair<int, int>(0, 3));
    edges.push_back(std::pair<int, int>(2, 4));
    edges.push_back(std::pair<int, int>(1, 5));
    edges.push_back(std::pair<int, int>(0, 6));
    edges.push_back(std::pair<int, int>(3, 7));
    edges.push_back(std::pair<int, int>(6, 5));
    edges.push_back(std::pair<int, int>(5, 4));
    edges.push_back(std::pair<int, int>(4, 7));
    edges.push_back(std::pair<int, int>(7, 6));

    cv::Mat box = cv::Mat(8, 3, CV_32F, box_verts).clone();
    box = box * 100.0f;

    cv::Vec3f rotation((float) vec_rot.at<double>(0, 0), (float) vec_rot.at<double>(1, 0),
                       (float) vec_rot.at<double>(2, 0));
    cv::Matx33f rot = euler2rotation_matrix(rotation);
    cv::Mat rot_box(8, 3, CV_32FC1);

    // Rotate the box
    rot_box = cv::Mat(rot) * cv::Mat_<float>(box.t());
    rot_box = rot_box.t();

    // Move the bounding box to head position
    for (int i = 0; i < 8; i++) {
        rot_box.at<float>(i, 0) += vec_trans.at<double>(0);
        rot_box.at<float>(i, 1) += vec_trans.at<double>(1);
        rot_box.at<float>(i, 2) += vec_trans.at<double>(2);
    }
    // draw the lines
    cv::Mat rot_box_proj(8, 3, CV_32F);
    project(rot_box_proj, rot_box);

    std::vector<std::pair<cv::Point2f, cv::Point2f> > lines;

    for (size_t i = 0; i < edges.size(); ++i) {
        cv::Mat_<float> begin;
        cv::Mat_<float> end;

        rot_box_proj.row(edges[i].first).copyTo(begin);
        rot_box_proj.row(edges[i].second).copyTo(end);

        cv::Point2f p1(begin.at<float>(0), begin.at<float>(1));
        cv::Point2f p2(end.at<float>(0), end.at<float>(1));

        lines.push_back(std::pair<cv::Point2f, cv::Point2f>(p1, p2));
    }
    box.release();
    rot_box.release();
    rot_box_proj.release();
    return lines;
}

void FaceEyeTrackingManager::eye_site_3d(FaceInfo *face_info, VPoint eye_site2d, VPoint3 &eye_site3d,
                                         VPoint &eye_front_site, cv::Mat &vec_rot, cv::Mat &vec_trans,
                                         bool is_right_eye) {
    cv::Vec3f rotation((float) vec_rot.at<double>(0, 0), (float) vec_rot.at<double>(1, 0),
                       (float) vec_rot.at<double>(2, 0));
    cv::Matx33f rot_vec = euler2rotation_matrix(rotation);

    float scale = 0.0;
    if (is_right_eye) {
        for (int i = FLM_42_R_EYEBROW_TOP_LEFT_CORNER; i < FLM_50_R_EYEBROW_LOWER_RIGHT_QUARTER; i++) {
            scale = scale + rot_vec(2, 0) * face_info->landmark3D68[i].x +
                    rot_vec(2, 1) * face_info->landmark3D68[i].y
                    + rot_vec(2, 2) * face_info->landmark3D68[i].z + vec_trans.at<double>(2);
        }
        scale /= 6;
    } else {
        for (int i = FLM_50_R_EYEBROW_LOWER_RIGHT_QUARTER; i < FLM_41_L_EYEBROW_LOWER_RIGHT_CORNER; i++) {
            scale = scale + rot_vec(2, 0) * face_info->landmark3D68[i].x +
                    rot_vec(2, 1) * face_info->landmark3D68[i].y
                    + rot_vec(2, 2) * face_info->landmark3D68[i].z + vec_trans.at<double>(2);
        }
        scale /= 6;
    }

    cv::Mat eye_site_camd = cv::Mat(3, 1, CV_32FC1);//相机坐标系
    eye_site_camd.at<float>(0, 0) = scale * (eye_site2d.x - _cx) / _fx;
    eye_site_camd.at<float>(1, 0) = scale * (eye_site2d.y - _cy) / _fy;
    eye_site_camd.at<float>(2, 0) = scale;

    cv::Mat eye_site_w = cv::Mat(3, 1, CV_64FC1);//世界坐标系
    eye_site_w.at<double>(0, 0) = eye_site_camd.at<float>(0, 0) - vec_trans.at<double>(0);
    eye_site_w.at<double>(1, 0) = eye_site_camd.at<float>(1, 0) - vec_trans.at<double>(1);
    eye_site_w.at<double>(2, 0) = eye_site_camd.at<float>(2, 0) - vec_trans.at<double>(2);
    eye_site_w = cv::Mat(rot_vec.t()) * cv::Mat_<float>(eye_site_w);
    eye_site3d.x = eye_site_w.at<float>(0, 0);
    eye_site3d.y = eye_site_w.at<float>(1, 0);
    eye_site3d.z = eye_site_w.at<float>(2, 0);

    //end Point
    cv::Mat a_eye_site_front = cv::Mat(4, 1, CV_64FC1);//世界坐标系
    a_eye_site_front.at<double>(0, 0) = eye_site3d.x;
    a_eye_site_front.at<double>(1, 0) = eye_site3d.y;
    a_eye_site_front.at<double>(2, 0) = eye_site3d.z - 100.0f;
    a_eye_site_front.at<double>(3, 0) = 1;

    cv::Mat eye_site_front_camd = cv::Mat(3, 1, CV_64FC1);//相机坐标系
    eye_site_front_camd = cv::Mat(rot_vec) * cv::Mat_<float>(eye_site_front_camd);
    eye_site_front_camd.at<float>(0, 0) += vec_trans.at<double>(0);
    eye_site_front_camd.at<float>(1, 0) += vec_trans.at<double>(1);
    eye_site_front_camd.at<float>(2, 0) += vec_trans.at<double>(2);

    float image_scale = 0.0;
    eye_front_site.x =
            eye_site_front_camd.at<float>(0, 0) * _fx + _cx * eye_site_front_camd.at<float>(2, 0);
    eye_front_site.y =
            eye_site_front_camd.at<float>(1, 0) * _fy + _cy * eye_site_front_camd.at<float>(2, 0);
    image_scale = eye_site_front_camd.at<float>(2, 0);
    eye_front_site.x /= image_scale;
    eye_front_site.y /= image_scale;

    eye_site_w.release();
    a_eye_site_front.release();
    eye_site_front_camd.release();
}

void FaceEyeTrackingManager::reset_two_points(VPoint &p1, VPoint &p2) {
    int temp = 0;
    if (p2.x < p1.x) {
        temp = p1.x;
        p1.x = p2.x;
        p2.x = temp;
        temp = p1.y;
        p1.y = p2.y;
        p2.y = temp;
    }
}

//根据点位置画直线
void FaceEyeTrackingManager::drawline(VPoint p1, VPoint p2, VEyeInfo roi_eye, int value, char *mask) {
    reset_two_points(p1, p2);
    int x = p1.x;
    int y = p1.y;
    int dy = p2.y - p1.y;
    int dx = p2.x - p1.x;
    int p = 0;
    if (p1.x == p2.x || p1.y == p2.y || abs(dx) == abs(dy)) {
        int incrementx = 0;
        int incrementy = 0;
        int step = 0;
        if (abs(dx) >= abs(dy)) {
            step = abs(dx);
        } else {
            step = abs(dy);
        }
        incrementx = dx / step;
        incrementy = dy / step;
        set_pixel(x, y, roi_eye, value, mask);
        for (int k = 0; k < step; k++) {
            x += incrementx;
            y += incrementy;
            set_pixel(x, y, roi_eye, value, mask);
        }
    } else {
        float k = float(dy) / float(dx);
        if (fabs(k) < 1) {
            p = 2 * abs(dy) - dx;
            set_pixel(x, y, roi_eye, value, mask);
            while (x < p2.x) {
                ++x;
                if (p < 0) {
                    p += 2 * abs(dy);
                } else {
                    if (dy > 0) {
                        y++;
                    } else {
                        y--;
                    }
                    p += 2 * abs(dy) - 2 * dx;
                }
                set_pixel(x, y, roi_eye, value, mask);
            }
        } else {
            reset_two_points(p1, p2);
            p = 2 * abs(dx) - dy;
            set_pixel(x, y, roi_eye, value, mask);
            while (y < p2.y) {
                ++y;
                if (p < 0) {
                    p += 2 * abs(dx);
                } else {
                    if (dx > 0) {
                        x++;
                    } else {
                        x--;
                    }
                    p += 2 * abs(dx) - 2 * dy;
                }
                set_pixel(x, y, roi_eye, value, mask);
            }

        }
    }
}

bool FaceEyeTrackingManager::set_pixel(int x, int y, VEyeInfo roi_eye, int value, char *mask) {
    int index = x - roi_eye.x + (y - roi_eye.y) * roi_eye.width;
    mask[index] = value;
    return true;
}

void FaceEyeTrackingManager::project(cv::Mat &rot_box_proj, cv::Mat rot_box) {
    for (int i = 0; i < 8; i++) {
        rot_box_proj.at<float>(i, 0) = rot_box.at<float>(i, 0) * _fx + _cx * rot_box.at<float>(i, 2);
        rot_box_proj.at<float>(i, 1) = rot_box.at<float>(i, 1) * _fy + _cy * rot_box.at<float>(i, 2);
        rot_box_proj.at<float>(i, 2) = rot_box.at<float>(i, 2);
        rot_box_proj.at<float>(i, 0) /= rot_box_proj.at<float>(i, 2);
        rot_box_proj.at<float>(i, 1) /= rot_box_proj.at<float>(i, 2);
    }
}


// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceEyeTrackingManager", ABILITY_FACE_EYE_TRACKING,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceEyeTrackingManager>());
});

} // namespace aura::vision

#endif