#include "vision/util/VisionUtil.h"

#include "opencv2/opencv.hpp"
#include "vision/util/ImageUtil.h"
#include "vision/util/log.h"
#include <cmath>
#include <vector>

namespace aura::vision {

static const char* TAG = "VisionUtil";

void VisionUtil::yuv_to_rgb(const void* src, void* dst, int w, int h) {
    ImageUtil::cvt_color_yuv2rgb((unsigned char*)src, (int*)dst, w, h);
}

void VisionUtil::save_image(const char* path, const void* frame, int w, int h) {
    if (path == nullptr) {
        VLOGE(TAG, "save_image(...) path is nullptr, save failed!");
        return;
    }

    if (frame == nullptr || (w <= 0 && h <= 0)) {
        VLOGE(TAG, "save_image(...) image is invalid!");
        return;
    }

    cv::Mat image(h * 3 / 2, w, CV_8UC1, (char*)frame);
    if (image.empty()) {
        return;
    }

    cv::Mat bgr;
    ImageUtil::cvt_color(image, bgr, cv::COLOR_YUV2BGR_NV21);
#ifdef WITH_OCV_HIGHGUI
    cv::imwrite(path, bgr);
#endif
}

void VisionUtil::euler_to_rotation_matrix(float yaw, float pitch, float roll, float *vector_angle) {
    static const float coordinate_list[4][3] = {{0,   0,   0},
                                                {100, 0,   0},
                                                {0,   100, 0},
                                                {0,   0,   100}};

    float sin_yaw = sin(yaw);
    float sin_pitch = sin(pitch);
    float sin_roll = sin(roll);
    float cos_yaw = cos(yaw);
    float cos_pitch = cos(pitch);
    float cos_roll = cos(roll);

    float rotation_matrix[3][3] = {};
    rotation_matrix[0][0] = cos_pitch * cos_roll;
    rotation_matrix[0][1] = -cos_pitch * sin_roll;
    rotation_matrix[0][2] = sin_pitch;
    rotation_matrix[1][0] = cos_yaw * sin_roll + cos_roll * sin_yaw * sin_pitch;
    rotation_matrix[1][1] = cos_yaw * cos_roll - sin_yaw * sin_pitch * sin_roll;
    rotation_matrix[1][2] = -cos_pitch * sin_yaw;
    rotation_matrix[2][0] = sin_yaw * sin_roll - cos_yaw * cos_roll * sin_pitch;
    rotation_matrix[2][1] = cos_roll * sin_yaw + cos_yaw * sin_pitch * sin_roll;
    rotation_matrix[2][2] = cos_yaw * cos_pitch;

    float coordinate_dest[4][3] = {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            coordinate_dest[j][i] = rotation_matrix[i][0] * coordinate_list[j][0] +
                                    rotation_matrix[i][1] * coordinate_list[j][1] +
                                    rotation_matrix[i][2] * coordinate_list[j][2];
        }
    }

    vector_angle[0] = coordinate_dest[1][0];
    vector_angle[1] = coordinate_dest[2][1];
    vector_angle[2] = coordinate_dest[3][2];
}

void VisionUtil::face_angle_optimal(const VAngle& angle_a, const VAngle& angle_b, std::vector<float>& output) {
    float vector_angle_a[3] = {};
    euler_to_rotation_matrix((float)(-angle_a.yaw * M_PI / 180.0f), (float)(-angle_a.pitch * M_PI / 180.0f),
                             (float)(angle_a.roll * M_PI / 180.0f), vector_angle_a);

    float vector_angle_b[3] = {};
    euler_to_rotation_matrix((float)(-angle_b.yaw * M_PI / 180.0f), (float)(-angle_b.pitch * M_PI / 180.0f),
            (float)(angle_b.roll * M_PI / 180.0f), vector_angle_b);

    float num = vector_angle_a[0] * vector_angle_b[0] + vector_angle_a[1] * vector_angle_b[1] +
                vector_angle_a[2] * vector_angle_b[2];

    auto pow_a = static_cast<float>(pow(vector_angle_a[0], 2) +
            pow(vector_angle_a[1], 2) +
            pow(vector_angle_a[2], 2));
    auto pow_b = static_cast<float>(pow(vector_angle_b[0], 2) +
            pow(vector_angle_b[1], 2) +
            pow(vector_angle_b[2], 2));
    auto denom = static_cast<float>(pow(pow_a, 0.5) * pow(pow_b, 0.5));

    float cosv = num / denom;

    output.resize(2);
    output[0] = 1.f - (acos(cosv) / M_PI);
    output[1] = (asin(sqrt(1 - cosv * cosv))) * 180 / M_PI;
}

} // namespace