//
// Created by Frewen.Wang on 2022/8/7.
//
#pragma once

#include "aura/utils/core/macro.h"

namespace aura::aura_cv {
/**
 * @brief AuraCV 的自定义坐标点类型
 */
class AURA_PUBLIC VPoint {
public:
    VPoint(float _x, float _y) {
        x = _x;
        y = _y;
    }

    VPoint() {
        x = 0.0F;
        y = 0.0F;
    }

    /**
     * @brief 对象拷贝
     * @param info  传入的引用
     */
    void copy(const VPoint &info);

    /**
     * @brief clear
     */
    void clear();

    friend VPoint operator+(const VPoint &p1, const VPoint &p2);

    friend VPoint operator-(const VPoint &p1, const VPoint &p2);

    /**
     * @brief 运算符重构
     * @param p
     */
    void operator+=(const VPoint &p);

    void operator-=(const VPoint &p);

    void operator/=(float val);

    float x;
    float y;
};

/**
 * @brief AuraCV 的自定义坐标区域类型
 */
class AURA_PUBLIC VRect {
public:
    float left;
    float top;
    float right;
    float bottom;

    VRect() : left(0.f), top(0.f), right(0.f), bottom(0.f) {}

    VRect(float _left, float _top, float _right, float _bottom)
        : left(_left), top(_top), right(_right), bottom(_bottom) {}

    /**
     * @brief 设置坐标区域的点
     * @param left
     * @param top
     * @param right
     * @param bottom
     */
    void set(float left, float top, float right, float bottom);

    /**
     * @brief 获取区域的宽度
     * @return
     */
    float width() const;

    /**
     * @brief 获取区域的高度
     * @return
     */
    float height() const;

    /**
     * @brief
     * @param x
     * @param y
     * @return
     */
    bool contains(float x, float y);

    /**
     * 框数据的拷贝
     * @param rect  rect
     */
    void copy(const VRect &rect);

    /**
     * 清空数据
     */
    void clear();
};

/**
 * @brief AuraVision 的三维坐标点类型
 */
class AURA_PUBLIC VPoint3 {
public:
    VPoint3(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    VPoint3() {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    /**
     * @brief 资源拷贝
     * @param info
     */
    void copy(const VPoint3 &info);

    void clear();

    /**
     *@brief 设置 VPoint3 的值
     * @param _x
     * @param _y
     * @param _z
     */
    void setValue(float _x, float _y, float _z);

    float x;
    float y;
    float z;
};

class AURA_PUBLIC VAngle {
public:
    VAngle(float x, float y, float z) {
        yaw = x;
        pitch = y;
        roll = z;
    }

    VAngle() {
        yaw = 0.0F;
        pitch = 0.0F;
        roll = 0.0F;
    }

    void copy(const VAngle &info);

    void clear();

    float yaw;
    float pitch;
    float roll;
};


} // namespace aura::aura_cv
