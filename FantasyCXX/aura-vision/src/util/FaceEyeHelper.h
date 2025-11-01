
#ifdef BUILD_EXPERIMENTAL

#pragma once

#include <opencv2/opencv.hpp>
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"

namespace aura::vision {
    class FaceEyeHelper {
    public:
        static FaceEyeHelper *instance();
        FaceEyeHelper();
        ~FaceEyeHelper();

        void detect(VisionRequest *request, VisionResult *result, RtConfig* rtConfig);

        int get_eye_rect(FaceInfo *face_info, VEyeInfo &eye_roi, int xlimit, int ylimit, bool is_right);

        bool connect_domain(unsigned char *src_img, int mask_width, int mask_height, uchar *dst_image);

        void find_margin(FaceInfo *face_info, VEyeInfo roi_eye, unsigned char *mask, bool is_right);

        bool eye_coord(uchar *eye_value, uchar *eye_mask, VEyeInfo roi_eye, VPoint &eye_sit,
                       float threshold_value);

        float get_eye_threshold(uchar *eye_patch, int width, int height);

        void get_eye_centroid(FaceInfo *face_info, VEyeInfo &eye_roi, uchar *eye_mask, uchar *eye_data,
                              bool is_right);

    private:
        int _width;
        int _height;
        float _delta_left_x;
        float _delta_left_y;
        float _delta_right_x;
        float _delta_right_y;
        float _delta_x;
        float _delta_y;
        int _eye_width;
        int _eye_height;
        float _min;
        int _size;
        float _sum_x;
        float _sum_y;
        int _offsets[9] = {0, 1, 6, 2, 3, 4, 7, 5, 0};

        std::vector<VPoint> _eye_points;
        VPoint _right_eye_site;
        VPoint _left_eye_site;
        FaceInfo *_face_info;

        uchar *_right_eye_margin;
        uchar *_left_eye_margin;
        uchar *_right_eye_mask;
        uchar *_left_eye_mask;

        PerfUtil *_perf;
    };
} // namespace aura::vision

#endif