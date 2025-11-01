//
// Created by Frewen.Wang on 2024/10/23.
//
#pragma once

#include "aura/aad/kalman_filter/BaseKalmanFilter.h"

namespace aura{
namespace aad{

/**
 * Kalman Constant Velocity Filter
 * 恒定速度模型卡尔曼滤波器
 * 线性模型：CV模型是线性模型，因此可以直接应用标准卡尔曼滤波算法进行状态估计。
 * 状态变量：通常包括目标在二维平面上的位置（横坐标、纵坐标）、速度（线速度，方向通常为与x轴夹角，逆时针为正）等。
 * 			在某些情况下，可能还包括与x轴夹角的余弦值和正弦值，以减少计算中的三角函数运算。
 * 运动特性：目标以恒定的速度沿某一方向直线运动，不考虑加速度的变化。
 * 状态转移方程:
 * 在CV模型中，状态转移方程可以表示为目标的当前状态如何根据其前一时刻的状态以及系统噪声进行演变。具体形式取决于状态变量的选择，但通常包括位置和速度的更新。
 * 应用场景
 * 适用于目标运动速度变化不大，且运动轨迹接近直线的场景。
 * 例如，在低动态环境中的车辆追踪、无人机导航等。
 */
class KalmanCVFilter : public BaseKalmanFilter {

public:
    KalmanCVFilter();
    
    ~KalmanCVFilter();
    
    bool Init(const Eigen::VectorXd &initial_belief_states, const Eigen::MatrixXd &initial_uncertainty);
    
    // @brief predict the current state and uncertainty of system
    // @params[IN] transform_matrix: transform the state from the
    //             pre moment to current moment
    // @params[IN] env_uncertainty_matrix: the uncertainty brought by
    //             the environment when predict.
    /**
     * @brief predict the current state and uncertainty of system
     * @param delta_t
     * @return
     */
    bool predict(const double &delta_t);
    
    /**
     * @brief use the current observation to correct the predict
     * 使用当前最新的观测类更新预测结果
     * @params [IN]  cur_observation
     * @params [IN]  cur_observation_uncertainty
     * @params [IN]  motion_constraint_valid_flag
     * @return
     */
    bool update(const Eigen::VectorXd &cur_observation,
                const Eigen::MatrixXd &cur_observation_uncertainty,
                const bool motion_constraint_valid_flag);
    
    /**
     * @brief get the system states
     * @return
     */
    Eigen::VectorXd getStates() const override;
  
};

} // aura
} // aura_asd
