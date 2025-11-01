
#ifndef VISION_MATH_UTILS_H
#define VISION_MATH_UTILS_H

#include "vision/core/common/VStructs.h"

#include <cmath>
#include <vector>

namespace aura::vision {
class MathUtils {
public:

    /**
     * @brief conversion from angle to radians
     * @param angle 角度
     * @return 弧度
     */
    static float angle_to_radians(float &angle);

    /**
     * @brief conversion from radians to angle
     * @param radians 弧度
     * @return 角度
     */
    static float radians2Angle(float &radians);

    /**
     * @brief conversion from radians to angle of VPoint3
     * @param radiansX 弧度X
     * @param radiansY 弧度Y
     * @param radiansZ 弧度Z
     * @param point
     */
    static void radians2Angle(float &radiansX, float &radiansY, float &radiansZ, VPoint3 &point);

    /**
     * @brief 获取指定多个下标的人脸关键点的最大元素（x或y）
     * @param landmark
     * @param indices 用于比较的indices
     * @param elem 0表示比较x，1表示比较y
     * @return 最大元素值
     */
    static float get_max_landmark(const VPoint* landmark, const std::vector<int>& indices, int elem);

    /**
     * @brief 获取指定多个下标的人脸关键点的最小元素（x或y）
     * @param landmark
     * @param indices 用于比较的indices
     * @param elem 0表示比较x，1表示比较y
     * @return 最小元素值
     */
    static float get_min_landmark(const VPoint* landmark, const std::vector<int>& indices, int elem);

    /**
     * @brief 求解4x4矩阵的逆
     * @param src 输入矩阵
     * @param dst 逆矩阵
     * @return 是否可逆
     */
    static bool matrix_4x4_invert(const float *src, float *dst);

    /**
     * @brief 求解jie3x3矩阵的逆
     * @param src 输入矩阵
     * @param dst 逆矩阵
     * @return 是否可逆
     */
    static bool matrix_3x3_invert(const float *src, float *dst);

    /**
     * @brief 获取数组中最大元素的下标
     * @param data 输入数组
     * @param len 数组长度
     * @return 最大元素下标
     */
    static int argmax(float *data, int len);

    /**
     * @brief 指数归一化，输入数组的元素值范围将会被压缩到[0,1]之间，且其和为1
     * @param in 输入数组
     * @param out 输出的归一化数组
     * @param len 数组长度
     * @return 0表示成功，-1表示异常
     */
    static int softmax(float* in, float* out, int len);

    /**
     * @brief 将框修改为正方形，根据输入的左上，右下点从新计算框的位置
     * @param left_up 左上点
     * @param right_down 右下点
     * @param frameWidth 图像的宽
     * @param frameHeight 图像的高
     */
    static void change_box_to_square(VPoint& left_up, VPoint& right_down, VPoint& center, int frameWidth, int frameHeight);

    /**
     *方法命名要求大小写
     *@left_up 检测框左上角点坐标
     *@right_down 检测框右下角点坐标
     *@center 检测框中心点
     *@frameWidth 图像宽
     *@frameHeight 图像高
     *@ratio 检测框拓展的系数，比例
     */
    static void resizeExtendBox(VPoint& left_up, VPoint& right_down, VPoint& center, int frameWidth, int frameHeight, float ratio);

    /**
     * @brief sigmoid
     * @param x
     * @return sigmoid value
     */
    static float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    // 计算两个检测框的重合度（iou = Intersection Over Union）
    static float base_iou(const VRect& r1, const VRect& r2);

    /**
     * 计算两个点的欧拉距离
     * @param point1
     * @param point2
     * @return
     */
    static float eulerDistance(const VPoint &point1, const VPoint &point2);

private:
    /**
     * @brief SSE加速的4x4矩阵求逆
     * @param src 输入矩阵
     * @param dst 逆矩阵
     * @return 是否成功
     */
    static bool matrix_4x4_invert_sse(const float *src, float *dst);
};
} // namespace aura::vision

#endif //VISION_MATH_UTILS_H
