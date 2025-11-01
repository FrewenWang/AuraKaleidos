
#pragma once

#include <string>
#include <algorithm>
#include "Eigen/Core"

struct KalmanFilterInitOptions {
	//  int state_num = 0;
	//  int measurement_num = 0;f
	//  int control_num = 0;
};

namespace aura {
namespace aura_asd {

/**
 * 平方根卡尔曼滤波器（Square Root Kalman Filter）是一种改进的卡尔曼滤波算法，它通过维护协方差矩阵的平方根形式来提高数值稳定性和精度。
 * 与传统卡尔曼滤波器相比，平方根卡尔曼滤波器具有以下几个关键特点：
 *
 * 1. 数值稳定性：平方根卡尔曼滤波器通过直接操作协方差矩阵的平方根（通常是一个下三角矩阵，如Cholesky分解）来避免可能的数值不稳定性问题。
 * 在计算过程中，协方差矩阵可能会因为舍入误差而导致非正定，从而影响滤波器的性能。使用平方根形式可以更好地保持正定性。
 * 2. 精度提高：由于维护协方差矩阵的平方根形式，平方根卡尔曼滤波器能够减少舍入误差的累积，提高计算的精度。这在处理高维度状态空间或者长时间运行的情况下尤为重要。
 * 3. 稳定性增强：平方根形式在更新过程中，可以更好地保持协方差矩阵的正定性和对称性，增强滤波器的稳定性。
 *
 * 平方根卡尔曼滤波器的实现通常涉及以下步骤：
 * 1. 初始化：初始化状态估计和协方差矩阵的平方根。一开始可能需要通过Cholesky分解将协方差矩阵分解成平方根形式。
 * 2. 预测步骤：使用系统模型预测下一个状态，并更新协方差矩阵平方根。
 * 3. 更新步骤：根据观测值更新状态估计和协方差矩阵的平方根。通常使用QR分解等数值方法来实现更新。
 *
 * 平方根卡尔曼滤波器在处理非线性系统时，可以与扩展卡尔曼滤波器（EKF）或无迹卡尔曼滤波器（UKF）结合使用，
 * 形成平方根扩展卡尔曼滤波器（Square Root EKF）或平方根无迹卡尔曼滤波器（Square Root UKF）。
 * 这种滤波器在导航、跟踪、信号处理等领域有着广泛的应用，尤其是在要求高精度和可靠性的场合。
 **/
class SRKalmanFilter {
public:
	SRKalmanFilter() = default;
	
	~SRKalmanFilter() = default;
	
	void Init();
	
	void Predict();
	
	void Correct(const Eigen::VectorXd &measurement_vector);
	
	void SymmetricCorrect(const Eigen::VectorXd &measurement_vector);
	
	void SetFeedBackRatio(Eigen::VectorXd &delta_vector);
	
	std::string ToString() const;
	
	inline const Eigen::VectorXd &GetStateVector() const { return state_vector_; }
	
	inline const Eigen::VectorXd &GetResidualVector() const { return residual_vector_; }
	
	std::string Name() const { return "SRKalmanFilter"; }

public:
	Eigen::VectorXd state_vector_;                     // Xk
	Eigen::MatrixXd state_trans_matrix_;               // Fk
	Eigen::MatrixXd control_matrix_;                   // Bk
	Eigen::VectorXd control_vector_;                   // uk
	Eigen::MatrixXd covariance_matrix_;                // Pk
	Eigen::MatrixXd process_noise_model_matrix_;       // Tk
	Eigen::MatrixXd process_noise_covariance_matrix_;  // Qk
	Eigen::MatrixXd process_noise_matrix_;             // Tk * Qk * Tk^T
	Eigen::VectorXd measurement_vector_;               // Zk
	Eigen::MatrixXd measurement_trans_matrix_;         // Hk
	Eigen::MatrixXd measurement_noise_matrix_;         // Rk
	Eigen::MatrixXd kalman_gain_matrix_;               // K'
	Eigen::VectorXd residual_vector_;                  // Zk - Hk * Xk
	Eigen::VectorXd estimate_measurement_vector_;      // Hk * Xk
	Eigen::VectorXd feed_back_thresh_;
	bool particleFeedBack = false;
	Eigen::MatrixXd sr_covariance_matrix_;                // Pk^1/2, lower triangular matrix
	Eigen::MatrixXd sr_process_noise_covariance_matrix_;  // Qk^1/2, diagonal matrix
};

}  // namespace aura
}  // namespace aura_asd
