#ifndef VISION_STRUCT_H
#define VISION_STRUCT_H

#include "vision/core/common/VMacro.h"
#include <sstream>

namespace aura::vision {

class VA_PUBLIC VPoint {
public:
    VPoint(float _x, float _y) {
        x = _x;
        y = _y;
    }

    VPoint() {
        x = 0.0F;
        y = 0.0F;
    }

    void copy(const VPoint &info);

    void clear();

    friend VPoint operator+(const VPoint &p1, const VPoint &p2);

    friend VPoint operator-(const VPoint &p1, const VPoint &p2);

    void operator+=(const VPoint &p);

    void operator-=(const VPoint &p);

    void operator/=(float val);

    /**
     *@brief 设置 VPoint 的值
     * @param _x
     * @param _y
     */
    void setValue(float _x, float _y);

    void toString(std::stringstream &ss) const;

    float x;
    float y;
};

class VA_PUBLIC VPoint3 {
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

    void copy(const VPoint3 &info);

    void clear();

    void toString(std::stringstream &ss) const;

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

class VA_PUBLIC VAngle {
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

    void toString(std::stringstream &ss) const;

    float yaw;
    float pitch;
    float roll;
};

class VA_PUBLIC VEyeInfo {
public:
    VEyeInfo(float _x, float _y, float _width, float _height) : x(_x), y(_y), width(_width), height(_height) {
    }

    VEyeInfo() : x(0), y(0), width(0), height(0) {
    }

    void copy(const VEyeInfo &info);

    void clear();

    void toString(std::stringstream &ss) const;

    float x, y, width, height;
    VPoint _eye_center;
    VPoint _eye_centroid;
};

class VA_PUBLIC VMatrix {
public:
    VMatrix(float _x, float _y, float _z) {
        x = _x;
        y = _y;
        z = _z;
    }

    VMatrix() {
        x = 0.0F;
        y = 0.0F;
        z = 0.0F;
    }

    void copy(const VMatrix &info);

    void clear();

    void toString(std::stringstream &ss) const;

    float x;
    float y;
    float z;
};

struct VA_PUBLIC VRect {
    float left;
    float top;
    float right;
    float bottom;

    VRect() : left(0.f), top(0.f), right(0.f), bottom(0.f) { }

    VRect(float _left, float _top, float _right, float _bottom)
            : left(_left), top(_top), right(_right), bottom(_bottom) { }

    void set(float left, float top, float right, float bottom);

    float width() const;

    float height() const;

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

    void toString(std::stringstream &ss) const;
};

// 简单的size
struct VA_PUBLIC SimpleSize {
public:
    float width;
    float height;
};
// 坐标极值
struct VA_PUBLIC ExtreSize {
    int x_min;
    int y_min;
    int x_max;
    int y_max;
};
// 记录索引和值
struct VA_PUBLIC IndexValue {
    int index;
    float value;
};

enum model_class_type {
    param = 1,
    bin = 2,
    txt = 3
};

enum VSlidingState {
    NON,     // 没有检测到
    START,   // 开始
    ONGOING, // 持续
    END      // 结束
};

struct VA_PUBLIC VState {
    int state; // 0-没有监测到, 1-开始，2-持续，3-中断
    int continue_time; // 持续的时间
    int trigger_count; // 触发的次数

    VState() { clear(); }

    void clear();

    void copy(const VState &info);
    void toString(std::stringstream &ss) const;
};

struct VA_PUBLIC VisGesture {
    int label;
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
};

// 均值算法
enum NORMAL_ALG {
    MUL, DIV
};

} // namespace vision

#endif // VISION_STRUCT_H











