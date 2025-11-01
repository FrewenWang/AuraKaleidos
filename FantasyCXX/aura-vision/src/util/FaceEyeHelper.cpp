//
// Created by Li,Wendong on 2018-12-30.
//

#ifdef BUILD_EXPERIMENTAL

#include "FaceEyeHelper.h"
#include "vision/core/common/VStructs.h"
#include "vision/util/log.h"

namespace aura::vision {
    using namespace std;
    const static string TAG = "FaceEyeHelper";
    static FaceEyeHelper *_s_instance;

    FaceEyeHelper *FaceEyeHelper::instance() {
        if (_s_instance == NULL) {
            _s_instance = new FaceEyeHelper();
        }
        return _s_instance;
    }

    FaceEyeHelper::FaceEyeHelper() {
    }

    FaceEyeHelper::~FaceEyeHelper() {
        if (_right_eye_margin != NULL) {
            delete[] _right_eye_margin;
            _right_eye_margin = NULL;
        }
        if (_right_eye_mask != NULL) {
            delete[] _right_eye_mask;
            _right_eye_mask = NULL;
        }
        if (_left_eye_margin != NULL) {
            delete[] _left_eye_margin;
            _left_eye_margin = NULL;
        }
        if (_left_eye_mask != NULL) {
            delete[] _left_eye_mask;
            _left_eye_mask = NULL;
        }
        if (_face_info != NULL) {
            delete _face_info;
            _face_info = NULL;
        }
    }

    void FaceEyeHelper::detect(VisionRequest *request, VisionResult *result,RtConfig *rtConfig) {
        VEyeInfo right_eye_roi;
        VEyeInfo left_eye_roi;
        auto* face_request = request->getFaceRequest();
        _width = face_request->width;
        _height = face_request->height;
        auto* face_result = result->getFaceResult();
        _perf = result->getPerfUtil();

        for (int i = 0; i < face_request->faceCount; i++) {
            _face_info = face_result->faceInfos[i];

            // 因为在 per_detect 做过判断，所以在单人脸时不会为 null
            if (face_request->faceCount > 1) {
                // 未检测到人脸时,不做处理
                if (_face_info == NULL || _face_info->noFace()) {
                    continue;
                }
            }

            PERF_TICK(_perf, "face_eye_1_get_eye_rect");
            // 1, 获取眼睛 bounding box
            get_eye_rect(_face_info, right_eye_roi, _width, _height, true);
            get_eye_rect(_face_info, left_eye_roi, _width, _height, false);
            PERF_TOCK(_perf, "face_eye_1_get_eye_rect");

            if (right_eye_roi.width <= 0 || right_eye_roi.height <= 0 ||
                left_eye_roi.width <= 0 || left_eye_roi.height <= 0) {
                continue;
            }

            PERF_TICK(_perf, "face_eye_2_get_eye_margin");
            // 2, 获取眼睛轮廓
            int right_len = int(right_eye_roi.width * right_eye_roi.height);
            _right_eye_margin = new uchar[right_len];
            memset(_right_eye_margin, 0, right_len);
            find_margin(_face_info, right_eye_roi, _right_eye_margin, true);

            int left_len = int(left_eye_roi.width * left_eye_roi.height);
            _left_eye_margin = new uchar[left_len];
            memset(_left_eye_margin, 0, left_len);
            find_margin(_face_info, left_eye_roi, _left_eye_margin, false);
            PERF_TOCK(_perf, "face_eye_2_get_eye_margin");

            PERF_TICK(_perf, "face_eye_3_connect_eye_domain");
            // 3, 眼睛扣图
            _right_eye_mask = new uchar[right_len];
            memset(_right_eye_mask, 255, right_len);
            connect_domain(_right_eye_margin, (int) right_eye_roi.width, (int) right_eye_roi.height, _right_eye_mask);

            _left_eye_mask = new uchar[left_len];
            memset(_left_eye_mask, 255, left_len);
            connect_domain(_left_eye_margin, (int) left_eye_roi.width, (int) left_eye_roi.height, _left_eye_mask);
            PERF_TOCK(_perf, "face_eye_3_connect_eye_domain");

            cv::Mat face_img(_height, _width, CV_8UC1, face_request->frame);
            cv::Mat right_eye_img;
            cv::Mat left_eye_img;
            cv::Mat right_eye_patch;
            cv::Mat left_eye_patch;

            PERF_TICK(_perf, "face_eye_4_bitwise_and");
            face_img(cv::Rect((int) right_eye_roi.x,
                              (int) right_eye_roi.y,
                              (int) right_eye_roi.width,
                              (int) right_eye_roi.height))
                    .copyTo(right_eye_img);
            cv::Mat right_eye_mask_mat((int) right_eye_roi.height, (int) right_eye_roi.width, CV_8UC1, _right_eye_mask);
            // 对图像每个像素值进行二进制 "与" 操作
            cv::bitwise_and(right_eye_img, right_eye_img, right_eye_patch, right_eye_mask_mat);

            face_img(cv::Rect((int) left_eye_roi.x,
                              (int) left_eye_roi.y,
                              (int) left_eye_roi.width,
                              (int) left_eye_roi.height))
                    .copyTo(left_eye_img);
            cv::Mat left_eye_mask_mat((int) left_eye_roi.height, (int) left_eye_roi.width, CV_8UC1, _left_eye_mask);
            cv::bitwise_and(left_eye_img, left_eye_img, left_eye_patch, left_eye_mask_mat);
            PERF_TOCK(_perf, "face_eye_4_bitwise_and");

            // IR 根据瞳孔的中心点(质点)
            if (rtConfig->cameraLightType == CAMERA_LIGHT_TYPE_IR) {
                PERF_TICK(_perf, "face_eye_5_get_eye_centroid");
                get_eye_centroid(_face_info, right_eye_roi, _right_eye_mask, right_eye_patch.data, true);
                get_eye_centroid(_face_info, left_eye_roi, _left_eye_mask, left_eye_patch.data, false);
                PERF_TOCK(_perf, "face_eye_5_get_eye_centroid");
            } else {
                PERF_TICK(_perf, "face_eye_5_get_eye_threshold");
                // 4, 自适应阈值求解
                float right_thresh = get_eye_threshold(right_eye_patch.data,
                                                       (int) right_eye_roi.width,
                                                       (int) right_eye_roi.height);

                float left_thresh = get_eye_threshold(left_eye_patch.data,
                                                      (int) left_eye_roi.width,
                                                      (int) left_eye_roi.height);
                PERF_TOCK(_perf, "face_eye_5_get_eye_threshold");

                PERF_TICK(_perf, "face_eye_6_eye_coord");
                // 5, 眼球中心点2D的坐标
                eye_coord(right_eye_patch.data, _right_eye_mask, right_eye_roi, _right_eye_site, right_thresh);
                eye_coord(left_eye_patch.data, _left_eye_mask, left_eye_roi, _left_eye_site, left_thresh);
                PERF_TOCK(_perf, "face_eye_6_eye_coord");

                PERF_TICK(_perf, "face_eye_7_post");
                if (_left_eye_site.x && _right_eye_site.x) {
                    _delta_left_x = _left_eye_site.x - left_eye_roi._eye_center.x;
                    _delta_left_y = _left_eye_site.y - left_eye_roi._eye_center.y;

                    _delta_right_x = _right_eye_site.x - right_eye_roi._eye_center.x;
                    _delta_right_y = _right_eye_site.y - right_eye_roi._eye_center.y;

                    _delta_x = _delta_left_x + _delta_right_x;
                    _delta_y = _delta_left_y + _delta_right_y;

                    _left_eye_site.x = left_eye_roi._eye_center.x + _delta_x;
                    _left_eye_site.y = left_eye_roi._eye_center.y + _delta_y;
                    _face_info->eyeCentroidLeft.x = _left_eye_site.x;
                    _face_info->eyeCentroidLeft.y = _left_eye_site.y;

                    _right_eye_site.x = right_eye_roi._eye_center.x + _delta_x;
                    _right_eye_site.y = right_eye_roi._eye_center.y + _delta_y;
                    _face_info->eyeCentroidRight.x = _right_eye_site.x;
                    _face_info->eyeCentroidRight.y = _right_eye_site.y;
                }
                PERF_TOCK(_perf, "face_eye_7_post");
            }
        }
    }

    int FaceEyeHelper::get_eye_rect(FaceInfo *face_info, VEyeInfo &eye_roi,
                                    int xlimit, int ylimit, bool is_right) {
        VPoint *landmark106_2d = face_info->landmark2D106;
        int EYE_POINTS_START = 0;
        int EYE_POINTS_END = 0;
        if (!is_right) {
            EYE_POINTS_START = FLM_51_L_EYE_LEFT_CORNER;
            EYE_POINTS_END = FLM_60_L_EYE_CENTER;
        } else {
            EYE_POINTS_START = FLM_61_R_EYE_LEFT_CORNER;
            EYE_POINTS_END = FLM_70_R_EYE_CENTER;
        }

        int ymin = (int) landmark106_2d[EYE_POINTS_START].y;
        int ymax = ymin;
        int xmin = (int) landmark106_2d[EYE_POINTS_START].x;
        int xmax = xmin;
        eye_roi._eye_center.x = 0;
        eye_roi._eye_center.y = 0;

        for (int i = EYE_POINTS_START; i <= EYE_POINTS_END; ++i) {
            if (ymin > landmark106_2d[i].y) {
                ymin = (int) landmark106_2d[i].y;
            }
            if (ymax < landmark106_2d[i].y) {
                ymax = (int) landmark106_2d[i].y;
            }
            if (xmin > landmark106_2d[i].x) {
                xmin = (int) landmark106_2d[i].x;
            }
            if (xmax < landmark106_2d[i].x) {
                xmax = (int) landmark106_2d[i].x;
            }
        }

        for (int i = EYE_POINTS_START; i < EYE_POINTS_END - 1; ++i) {
            eye_roi._eye_center.x += landmark106_2d[i].x;
            eye_roi._eye_center.y += landmark106_2d[i].y;
        }

        xmin = xmin < 2 ? 2 : xmin;
        xmin = xmin > xlimit - 2 ? xlimit - 2 : xmin;
        xmax = xmax < 2 ? 2 : xmax;
        xmax = xmax > xlimit - 2 ? xlimit - 2 : xmax;

        ymin = ymin < 2 ? 2 : ymin;
        ymin = ymin > ylimit - 2 ? ylimit - 2 : ymin;
        ymax = ymax < 2 ? 2 : ymax;
        ymax = ymax > ylimit - 2 ? ylimit - 2 : ymax;

        eye_roi.x = xmin - 2;
        eye_roi.y = ymin - 2;
        eye_roi.height = ymax - ymin + 4;
        eye_roi.width = xmax - xmin + 4;
        eye_roi._eye_center.x /= 8;
        eye_roi._eye_center.y /= 8;

        return true;
    }

    void FaceEyeHelper::find_margin(FaceInfo *face_info, VEyeInfo roi_eye,
                                    unsigned char *mask, bool is_right) {

        int POINTS_START = 0;
        if (!is_right) {
            POINTS_START = FLM_51_L_EYE_LEFT_CORNER;
        } else {
            POINTS_START = FLM_61_R_EYE_LEFT_CORNER;
        }

        VPoint *landmark106 = face_info->landmark2D106;
        float offset_y = roi_eye.height * 0.1;
        float offset_x = roi_eye.width * 0.1;
        cv::Scalar color(255);
        cv::Mat image((int) roi_eye.height, (int) roi_eye.width, CV_8UC1, mask);

        if (is_right) {
            float right_offset_pt1_x[8] = {0, 0, 0, 0, -offset_x, 0, 0, 0};
            float right_offset_pt1_y[8] = {0, offset_y, offset_y, offset_y, 0, -offset_y, -offset_y, -offset_y};
            float right_offset_pt2_x[8] = {0, 0, 0, offset_x, 0, 0, 0, 0};
            float right_offset_pt2_y[8] = {offset_y, offset_y, offset_y, 0, -offset_y, -offset_y, -offset_y, 0};

            for (int i = 0; i < 8; ++i) {
                auto p1 = cv::Point2f(landmark106[POINTS_START + _offsets[i]].x - roi_eye.x + right_offset_pt1_x[i],
                                      landmark106[POINTS_START + _offsets[i]].y - roi_eye.y + right_offset_pt1_y[i]);
                auto p2 = cv::Point2f(landmark106[POINTS_START + _offsets[i + 1]].x - roi_eye.x + right_offset_pt2_x[i],
                                      landmark106[POINTS_START + _offsets[i + 1]].y - roi_eye.y +
                                      right_offset_pt2_y[i]);
#ifdef OPENCV2
                cv::line(image, p1, p2, color);
#endif
#ifdef OPENCV4

#endif
            }
        } else {
            float left_offset_pt1_x[8] = {offset_x, 0, 0, 0, 0, 0, 0, 0};
            float left_offset_pt1_y[8] = {0, offset_y, offset_y, offset_y, 0, -offset_y, -offset_y, -offset_y};
            float left_offset_pt2_x[8] = {0, 0, 0, 0, 0, 0, 0, offset_x};
            float left_offset_pt2_y[8] = {offset_y, offset_y, offset_y, 0, -offset_y, -offset_y, -offset_y, 0};

            for (int i = 0; i < 8; ++i) {
                auto p1 = cv::Point2f(landmark106[POINTS_START + _offsets[i]].x - roi_eye.x + left_offset_pt1_x[i],
                                      landmark106[POINTS_START + _offsets[i]].y - roi_eye.y + left_offset_pt1_y[i]);
                auto p2 = cv::Point2f(landmark106[POINTS_START + _offsets[i + 1]].x - roi_eye.x + left_offset_pt2_x[i],
                                      landmark106[POINTS_START + _offsets[i + 1]].y - roi_eye.y
                                      + left_offset_pt2_y[i]);
#ifdef OPENCV2
                cv::line(image, p1, p2, color);
#endif
#ifdef OPENCV4
#endif
            }
        }
    }

    //SrcImg输入图像Mask，只有边缘为255，其他为0；DstImage输出图像Label，初始化均为255，边缘为0，人眼部分为255
    bool FaceEyeHelper::connect_domain(unsigned char *src_img, int mask_width, int mask_height, uchar *dst_image) {
        const int domain[4] = {-mask_width, -1, +1, mask_width};
        std::vector<int> round;
        int i_current = 0;
        dst_image[i_current] = 0;
        round.push_back(i_current);
        while (!round.empty()) {
            for (int k = 0; k < 4; k++) {
                if (i_current + domain[k] > 0 && i_current + domain[k] < mask_width * mask_height) {
                    if (src_img[i_current + domain[k]] == 0 && dst_image[i_current + domain[k]] == 255) {
                        round.push_back(i_current + domain[k]);

                        dst_image[i_current + domain[k]] = 0;//标为0的元素都为连通域元素,连通域之外的元素为255
                    }
                }
            }
            if (!round.empty()) {
                i_current = round.back();
                round.pop_back();
            }
        }
        return true;
    }

    bool FaceEyeHelper::eye_coord(uchar *eye_value, uchar *eye_mask, VEyeInfo roi_eye, VPoint &eye_sit,
                                  float threshold_value) {
        eye_sit.x = 0;
        eye_sit.y = 0;
        float totalx = 0.0;
        float totaly = 0.0;
        float max_value = 0.0;
        float min_value = 255.0;

        for (int i = 0; i < roi_eye.width * roi_eye.height; i++) {
            if (eye_value[i] != 0 && eye_mask[i] != 0) {
                if (eye_value[i] > max_value) {
                    max_value = eye_value[i];
                }
                if (eye_value[i] < min_value) {
                    min_value = eye_value[i];
                }
            }
        }

        float value_width = max_value - min_value;
        for (int i = 0; i < roi_eye.width * roi_eye.height; i++) {
            if (eye_value[i] != 0 && eye_value[i] < threshold_value && eye_mask[i] != 0) {
                int x = i % int(roi_eye.width);
                int y = i / int(roi_eye.width);
                eye_sit.x = eye_sit.x + (float) (max_value - eye_value[i]) / value_width * x;
                eye_sit.y = eye_sit.y + (float) (max_value - eye_value[i]) / value_width * y;
                totalx = totalx + (float) (max_value - eye_value[i]) / value_width;
                totaly = totaly + (float) (max_value - eye_value[i]) / value_width;
            }
        }

        static const float thresh = 0.0000001f;
        if (totalx > thresh && totaly > thresh) {
            eye_sit.x /= totalx;
            eye_sit.y /= totaly;
            eye_sit.x = eye_sit.x + roi_eye.x;
            eye_sit.y = eye_sit.y + roi_eye.y;
            return true;
        } else {
            return false;
        }
    }

    float FaceEyeHelper::get_eye_threshold(uchar *eye_patch, int width, int height) {
        float eye_pixel = 0.f;
        int cnt = 0;

        int size = height * width;
        for (int i = 0; i < size; ++i) {
            if (eye_patch[i] != 0) {
                eye_pixel += eye_patch[i];
                cnt++;
            }
        }

        return (float) ((float) eye_pixel / cnt + 0.0000001f);
    }

    /**
     * 获取 IR 模式的眼睛瞳孔质点
     * @param face_info
     * @param eye_roi
     * @param eye_mask
     * @param eye_img
     */
    void FaceEyeHelper::get_eye_centroid(FaceInfo *face_info, VEyeInfo &eye_roi,
                                         uchar *eye_mask, uchar *data, bool is_right) {
        _min = 255;
        _eye_width = eye_roi.width;
        _eye_height = eye_roi.height;
        int index = 0;
        for (int i = 0; i < _eye_height; i++) {
            for (int j = 2; j < _eye_width - 2; j++) {
                index = i * _eye_width + j;

                if (eye_mask[index] != 0
                    // && data[index] < data[index - 1]
                    // && (data[index] < data[index + 1] || data[index] < data[index + 2])
                    && data[index] < _min) {
                    _min = data[index];
                }
            }
        }

        VPoint eye_point;
        for (int i = 0; i < _eye_height; i++) {
            for (int j = 2; j < _eye_width - 2; j++) {
                index = i * _eye_width + j;

                if (eye_mask[index] != 0
                    && data[index] < data[index - 1]
                    && (data[index] < data[index + 1] || data[index] < data[index + 2])
                    && data[index] <= _min + 2) {
                    eye_point.x = j;
                    eye_point.y = i;
                    _eye_points.push_back(eye_point);
                }
            }
        }

        _size = _eye_points.size();
        _sum_x = 0;
        _sum_y = 0;
        if (_size != 0) {
            for (int i = 0; i < _size; i++) {
                _sum_x += _eye_points[i].x;
                _sum_y += _eye_points[i].y;
            }
            // 清空动态数组元素
            _eye_points.clear();

            if (is_right) {
                face_info->eyeCentroidRight.x = _sum_x / _size + eye_roi.x;
                face_info->eyeCentroidRight.y = _sum_y / _size + eye_roi.y;

            } else {
                face_info->eyeCentroidLeft.x = _sum_x / _size + eye_roi.x;
                face_info->eyeCentroidLeft.y = _sum_y / _size + eye_roi.y;
            }
        }
    }

} // namespace aura::vision

#endif