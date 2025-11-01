#include "vision/core/bean/VehicleInfo.h"

void VehicleInfo::clear() {
    speed = 0.f;
    turningLamp = false;
    steeringWheelAngle = 0.f;
}

void VehicleInfo::setValue(float speed, bool turningLamp, float steeringWheelAngle) {
    this->speed = speed;
    this->turningLamp = turningLamp;
    this->steeringWheelAngle = steeringWheelAngle;
}
