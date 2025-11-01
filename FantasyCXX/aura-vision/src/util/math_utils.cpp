//
// Created by v_guojinlong on 2019-04-08.
//

#include "math_utils.h"
#include <algorithm>
#include <cmath>

#if __SSE__

#include <xmmintrin.h>

#endif

namespace aura::vision {
static const int FACE_LANDMARK_COUNT = 106;

static const char *TAG = "MathUtils";

float MathUtils::angle_to_radians(float &angle) {
    return static_cast<float>(angle * M_PI / 180);
}

float MathUtils::radians2Angle(float &radians) {
    return static_cast<float>(radians * 180 / M_PI);
}

void MathUtils::radians2Angle(float &radiansX, float &radiansY, float &radiansZ, VPoint3 &point) {
    point.setValue(radiansX * 180 / M_PI, radiansY * 180 / M_PI, radiansZ * 180 / M_PI);
}

float MathUtils::get_max_landmark(const VPoint *landmark, const std::vector<int> &indices, int elem) {
    if (!landmark) {
        return -1.f;
    }

    bool succ_flag = true;
    auto iter = std::max_element(indices.begin(), indices.end(), [&](const int &i1, const int &i2) {
        if (i1 >= FACE_LANDMARK_COUNT || i2 >= FACE_LANDMARK_COUNT || i1 < 0 || i2 < 0) {
            succ_flag = false;
            return true;
        }

        if (elem == 0) return landmark[i1].x < landmark[i2].x;
        else return landmark[i1].y < landmark[i2].y;
    });

    float res = -1.f;
    if (succ_flag) {
        res = (elem == 0 ? landmark[*iter].x : landmark[*iter].y);
    }

    return res;
}

float MathUtils::get_min_landmark(const VPoint *landmark, const std::vector<int> &indices, int elem) {
    if (!landmark) {
        return -1.f;
    }

    bool succ_flag = true;
    auto iter = std::max_element(indices.begin(), indices.end(), [&](const int &i1, const int &i2) {
        if (i1 >= FACE_LANDMARK_COUNT || i2 >= FACE_LANDMARK_COUNT || i1 < 0 || i2 < 0) {
            succ_flag = false;
            return true;
        }

        if (elem == 0) return landmark[i1].x > landmark[i2].x;
        else return landmark[i1].y > landmark[i2].y;
    });

    float res = -1.f;
    if (succ_flag) {
        res = (elem == 0 ? landmark[*iter].x : landmark[*iter].y);
    }

    return res;
}

int MathUtils::argmax(float *data, int len) {
    if (!data) {
        return -1;
    }
    return static_cast<int>(std::distance(data, std::max_element(data, data + len)));
}

int MathUtils::softmax(float *in, float *out, int len) {
    if (!in || !out) {
        return -1;
    }

    float sum = 0;
    std::vector<float> exp_data(len);
    for (int i = 0; i < len; ++i) {
        float tmp = exp(in[i]);
        sum += tmp;
        exp_data[i] = tmp;
    }

    for (int i = 0; i < len; ++i) {
        if (fabs(sum) < 1e-6) {
            out[i] = in[i];
        } else {
            out[i] = exp_data[i] / sum;
        }
    }
    return 0;
}

void MathUtils::change_box_to_square(VPoint &left_up, VPoint &right_down, VPoint &center, int frameWidth, int frameHeight) {
    float rect_width = right_down.x - left_up.x;
    float rect_height = right_down.y - left_up.y;
    float max_side_half = std::max(rect_width, rect_height) / 2.0f;

    center.x = left_up.x + rect_width / 2.0f;
    center.y = left_up.y + rect_height / 2.0f;

    left_up.x = std::max(0.0f, center.x - max_side_half);
    left_up.y = std::max(0.0f, center.y - max_side_half);
    right_down.x = std::min((float) frameWidth, center.x + max_side_half);
    right_down.y = std::min((float) frameHeight, center.y + max_side_half);
}

void MathUtils::resizeExtendBox(VPoint &left_up, VPoint &right_down, VPoint &center, int frameWidth, int frameHeight, float ratio) {
    float rect_width = right_down.x - left_up.x;
    float rect_height = right_down.y - left_up.y;
    float rect_edge = std::max(rect_width, rect_height) * ratio;
    float rect_edge_half = rect_edge / 2.0f;
    center.x = (left_up.x + right_down.x) / 2.0f;
    center.y = (left_up.y + right_down.y) / 2.0f;
    left_up.x = std::max(0.0f,center.x - rect_edge_half);
    left_up.y = std::max(0.0f,center.y - rect_edge_half);
    right_down.x = std::max(0.0f, center.x + rect_edge_half);
    right_down.y = std::max(0.0f,center.y + rect_edge_half);
    left_up.x = std::min(left_up.x, (float) frameWidth);
    left_up.y = std::min(left_up.y, (float) frameHeight);
    right_down.x = std::min(right_down.x, (float) frameWidth);
    right_down.y = std::min(right_down.y, (float) frameHeight);
}

float MathUtils::base_iou(const VRect& r1, const VRect& r2) {
    auto x01 = r1.left;
    auto y01 = r1.top;
    auto x02 = r1.right;
    auto y02 = r1.bottom;

    auto x11 = r2.left;
    auto y11 = r2.top;
    auto x12 = r2.right;
    auto y12 = r2.bottom;

    auto dist_center_x = std::fabs((x01 + x02) / 2.f - (x11 + x12) / 2.f);
    auto dist_center_y = std::fabs((y01 + y02) / 2.f - (y11 + y12) / 2.f);
    auto dist_sum_x = (std::fabs(x01 - x02) + std::fabs(x11 - x12)) / 2.f;
    auto dist_sum_y = (std::fabs(y01 - y02) + std::fabs(y11 - y12)) / 2.f;

    if (dist_center_x > dist_sum_x || dist_center_y > dist_sum_y) {
        return 0.f;
    }

    auto cols = std::min(x02, x12) - std::max(x01, x11);
    auto rows = std::min(y02, y12) - std::max(y01, y11);
    auto intersection = cols * rows;
    auto area1 = (x02 - x01) * (y02 - y01);
    auto area2 = (x12 - x11) * (y12 - y11);
    auto coincide = intersection / (area1 + area2 - intersection);
    // VLOGD(TAG, "iou check coincide:[%f]", coincide);
    return coincide;
}

bool MathUtils::matrix_3x3_invert(const float *src, float *dst) {
    // 计算伴随
    dst[0] = src[4] * src[8] - src[5] * src[7];
    dst[1] = src[2] * src[7] - src[1] * src[5];
    dst[2] = src[1] * src[5] - src[4] * src[2];

    dst[3] = src[5] * src[6] - src[3] * src[8];
    dst[4] = src[0] * src[8] - src[6] * src[2];
    dst[5] = src[3] * src[2] - src[0] * src[5];

    dst[6] = src[3] * src[7] - src[4] * src[6];
    dst[7] = src[1] * src[6] - src[0] * src[7];
    dst[8] = src[0] * src[4] - src[3] * src[1];

    // 计算行列式
    float det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[1];
    if (fabs(det) < 1e-6) {
        return false;
    }

    // 计算逆矩阵
    det = 1.0f / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;

    return true;
}

bool MathUtils::matrix_4x4_invert(const float *src, float *dst) {
#if __SSE__
    return matrix_4x4_invert_sse(src, dst);
#else
    // 计算伴随
    dst[0] =
            +src[5] * src[10] * src[15]
            - src[5] * src[11] * src[14]
            - src[9] * src[6] * src[15]
            + src[9] * src[7] * src[14]
            + src[13] * src[6] * src[11]
            - src[13] * src[7] * src[10];

    dst[1] =
            -src[1] * src[10] * src[15]
            + src[1] * src[11] * src[14]
            + src[9] * src[2] * src[15]
            - src[9] * src[3] * src[14]
            - src[13] * src[2] * src[11]
            + src[13] * src[3] * src[10];

    dst[2] =
            +src[1] * src[6] * src[15]
            - src[1] * src[7] * src[14]
            - src[5] * src[2] * src[15]
            + src[5] * src[3] * src[14]
            + src[13] * src[2] * src[7]
            - src[13] * src[3] * src[6];

    dst[3] =
            -src[1] * src[6] * src[11]
            + src[1] * src[7] * src[10]
            + src[5] * src[2] * src[11]
            - src[5] * src[3] * src[10]
            - src[9] * src[2] * src[7]
            + src[9] * src[3] * src[6];

    dst[4] =
            -src[4] * src[10] * src[15]
            + src[4] * src[11] * src[14]
            + src[8] * src[6] * src[15]
            - src[8] * src[7] * src[14]
            - src[12] * src[6] * src[11]
            + src[12] * src[7] * src[10];

    dst[5] =
            +src[0] * src[10] * src[15]
            - src[0] * src[11] * src[14]
            - src[8] * src[2] * src[15]
            + src[8] * src[3] * src[14]
            + src[12] * src[2] * src[11]
            - src[12] * src[3] * src[10];

    dst[6] =
            -src[0] * src[6] * src[15]
            + src[0] * src[7] * src[14]
            + src[4] * src[2] * src[15]
            - src[4] * src[3] * src[14]
            - src[12] * src[2] * src[7]
            + src[12] * src[3] * src[6];

    dst[7] =
            +src[0] * src[6] * src[11]
            - src[0] * src[7] * src[10]
            - src[4] * src[2] * src[11]
            + src[4] * src[3] * src[10]
            + src[8] * src[2] * src[7]
            - src[8] * src[3] * src[6];

    dst[8] =
            +src[4] * src[9] * src[15]
            - src[4] * src[11] * src[13]
            - src[8] * src[5] * src[15]
            + src[8] * src[7] * src[13]
            + src[12] * src[5] * src[11]
            - src[12] * src[7] * src[9];

    dst[9] =
            -src[0] * src[9] * src[15]
            + src[0] * src[11] * src[13]
            + src[8] * src[1] * src[15]
            - src[8] * src[3] * src[13]
            - src[12] * src[1] * src[11]
            + src[12] * src[3] * src[9];

    dst[10] =
            +src[0] * src[5] * src[15]
            - src[0] * src[7] * src[13]
            - src[4] * src[1] * src[15]
            + src[4] * src[3] * src[13]
            + src[12] * src[1] * src[7]
            - src[12] * src[3] * src[5];

    dst[11] =
            -src[0] * src[5] * src[11]
            + src[0] * src[7] * src[9]
            + src[4] * src[1] * src[11]
            - src[4] * src[3] * src[9]
            - src[8] * src[1] * src[7]
            + src[8] * src[3] * src[5];

    dst[12] =
            -src[4] * src[9] * src[14]
            + src[4] * src[10] * src[13]
            + src[8] * src[5] * src[14]
            - src[8] * src[6] * src[13]
            - src[12] * src[5] * src[10]
            + src[12] * src[6] * src[9];

    dst[13] =
            +src[0] * src[9] * src[14]
            - src[0] * src[10] * src[13]
            - src[8] * src[1] * src[14]
            + src[8] * src[2] * src[13]
            + src[12] * src[1] * src[10]
            - src[12] * src[2] * src[9];

    dst[14] =
            -src[0] * src[5] * src[14]
            + src[0] * src[6] * src[13]
            + src[4] * src[1] * src[14]
            - src[4] * src[2] * src[13]
            - src[12] * src[1] * src[6]
            + src[12] * src[2] * src[5];

    dst[15] =
            +src[0] * src[5] * src[10]
            - src[0] * src[6] * src[9]
            - src[4] * src[1] * src[10]
            + src[4] * src[2] * src[9]
            + src[8] * src[1] * src[6]
            - src[8] * src[2] * src[5];

    /*计算行列式*/
    float det = src[0] * dst[0] + src[1] * dst[4] + src[2] * dst[8] + src[3] * dst[12];
    if (fabs(det) < 1e-6) {
        return false;
    }

    /*与行列式的倒数相乘*/
    det = 1.0f / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
    dst[4] *= det;
    dst[5] *= det;
    dst[6] *= det;
    dst[7] *= det;
    dst[8] *= det;
    dst[9] *= det;
    dst[10] *= det;
    dst[11] *= det;
    dst[12] *= det;
    dst[13] *= det;
    dst[14] *= det;
    dst[15] *= det;
    return true;
#endif
}

#if __SSE__

bool MathUtils::matrix_4x4_invert_sse(const float *src, float *dst) {
    __m128 minor0{}, minor1{}, minor2{}, minor3{};
    __m128 row0{}, row1{}, row2{}, row3{};
    __m128 det{}, tmp1{};

    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64 *) (src)), (__m64 *) (src + 4));
    row1 = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64 *) (src + 8)), (__m64 *) (src + 12));

    row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
    row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);

    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64 *) (src + 2)), (__m64 *) (src + 6));
    row3 = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64 *) (src + 10)), (__m64 *) (src + 14));

    row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
    row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);

    tmp1 = _mm_mul_ps(row2, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

    minor0 = _mm_mul_ps(row1, tmp1);
    minor1 = _mm_mul_ps(row0, tmp1);

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);

    tmp1 = _mm_mul_ps(row1, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

    minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3 = _mm_mul_ps(row0, tmp1);

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);

    tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2 = _mm_shuffle_ps(row2, row2, 0x4E);

    minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2 = _mm_mul_ps(row0, tmp1);

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);

    tmp1 = _mm_mul_ps(row0, row1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));

    tmp1 = _mm_mul_ps(row0, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));

    tmp1 = _mm_mul_ps(row0, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);

    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));

    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);

    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);

    det = _mm_mul_ps(row0, minor0);
    det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);

    tmp1 = _mm_rcp_ss(det);

    det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
    det = _mm_shuffle_ps(det, det, 0x00);

    minor0 = _mm_mul_ps(det, minor0);
    _mm_storel_pi((__m64 *) (dst), minor0);
    _mm_storeh_pi((__m64 *) (dst + 2), minor0);

    minor1 = _mm_mul_ps(det, minor1);
    _mm_storel_pi((__m64 *) (dst + 4), minor1);
    _mm_storeh_pi((__m64 *) (dst + 6), minor1);

    minor2 = _mm_mul_ps(det, minor2);
    _mm_storel_pi((__m64 *) (dst + 8), minor2);
    _mm_storeh_pi((__m64 *) (dst + 10), minor2);

    minor3 = _mm_mul_ps(det, minor3);
    _mm_storel_pi((__m64 *) (dst + 12), minor3);
    _mm_storeh_pi((__m64 *) (dst + 14), minor3);

    return true;
}
#endif

float MathUtils::eulerDistance(const VPoint &point1, const VPoint &point2) {
    return powf((powf(point1.x - point2.x, 2) + powf(point1.y - point2.y, 2)), 0.5);
}

}