//
// Created by Frewen.Wang on 2022/8/7.
//
#include "aura/cv/structs.h"

namespace aura::aura_cv {

void VPoint::copy(const VPoint &info) {
    x = info.x;
    y = info.y;
}

void VPoint::clear() {
    x = 0.0F;
    y = 0.0F;
}


VPoint operator+(const VPoint &p1, const VPoint &p2) {
    VPoint p(p1);
    p += p2;
    return p;
}

VPoint operator-(const VPoint &p1, const VPoint &p2) {
    VPoint p(p1);
    p -= p2;
    return p;
}

void VPoint::operator+=(const VPoint &p) {
    x += p.x;
    y += p.y;
}

void VPoint::operator-=(const VPoint &p) {
    x -= p.x;
    y -= p.y;
}

void VPoint::operator/=(float val) {
    x /= val;
    y /= val;
}

void VPoint3::copy(const VPoint3 &info) {
    x = info.x;
    y = info.y;
    z = info.z;
}

void VPoint3::clear() {
    x = 0.0F;
    y = 0.0F;
    z = 0.0F;
}

void VPoint3::setValue(float _x, float _y, float _z) {
    x = _x;
    y = _y;
    z = _z;
}

void VRect::set(float l, float t, float r, float b) {
    left = l;
    top = t;
    right = r;
    bottom = b;
}

float VRect::width() const {
    return right - left;
}

float VRect::height() const {
    return bottom - top;
}

bool VRect::contains(float x, float y) {
    return left < right && top < bottom  // check for empty first
           && x >= left && x < right && y >= top && y < bottom;
}

void VRect::copy(const VRect &rect) {
    left = rect.left;
    top = rect.top;
    right = rect.right;
    bottom = rect.bottom;
}

void VRect::clear() {
    left = 0.f;
    top = 0.f;
    right = 0.f;
    bottom = 0.f;
}

void VAngle::copy(const VAngle &info) {
    yaw = info.yaw;
    pitch = info.pitch;
    roll = info.roll;
}

void VAngle::clear() {
    yaw = 0.0;
    pitch = 0.0;
    roll = 0.0;
}


}