//
// Created by wangzhijiang on 3/20/23.
//
#pragma once

/**
 * 车辆相关信息数据
 */
class VehicleInfo {
public:
    /** 车辆车速 */
    float speed = 0.f;
    /** 转向灯开关状态：true:开启 false:关闭 */
    bool turningLamp = false;
    /** 方向盘的转向角度，正负皆可能 */
    float steeringWheelAngle = 0.f;

    /**
     * 数据清除、防止上帧脏数据影响结果
     */
    void clear();

    /**
     * 数据拷贝
     * @param speed 车辆车速
     * @param turningLamp  转向灯开关状态
     * @param steeringWheelAngle 方向盘的转向角度
     */
    void setValue(float speed, bool turningLamp, float steeringWheelAngle);
};
