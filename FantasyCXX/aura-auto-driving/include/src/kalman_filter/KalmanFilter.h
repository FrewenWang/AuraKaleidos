/**
 * @file KalmanFilter.h
 **/
#pragma once

#include "Eigen/Dense"

namespace aura {
namespace aura_asd {

class KalmanFilter {
public:
	/**
	 * 使用指定的矩阵创建卡尔曼滤波器
	 * @param A   	系统动力学矩阵  	System dynamics matrix
	 * @param B		系统输入矩阵		System control input matrix
	 * @param C		输出矩阵			System Measurement matrix
	 * @param Q		过程噪声协方差		Process noise covariance
	 * @param R		测量噪声协方差		Measurement noise covariance
	 * @param P		估计误差协方差		Estimate error covariance
	 */
	KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
				 Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd P
	);
	
	~KalmanFilter() = default;
	
	/**
	 * 使用默认状态的猜测初始化过滤器
	 */
	void init();
	
	/**
	 * 使用初始化状态的猜测初始化过滤器
	 * @param x0
	 */
	void init(const Eigen::VectorXd &x0);
	
	/**
	 * 根据控制输入更新预测
	 * @param u
	 */
	void predict(const Eigen::VectorXd &u);
	
	/**
	 * 根据控制输入更新预测
	 * @param u
	 */
	void predict();
	/**
	 *
	 * @param y
	 */
	void update(const Eigen::VectorXd &y);
	
	const Eigen::VectorXd &predictedState() const;
	
	const Eigen::VectorXd &updatedState() const;
	
	void setPredictedState(const Eigen::VectorXd &x_hat_new);
	
	void setUpdatedState(const Eigen::VectorXd &x_hat);
	
	/**
	 * Update the dynamics matrix.
	 */
	void updateDynamics(const Eigen::MatrixXd A);
	
	/**
	* Update the output matrix.
	*/
	void updateOutput(const Eigen::MatrixXd C);

private:
	Eigen::MatrixXd _A;  // System dynamics matrix
	Eigen::MatrixXd _B;  // System control matrix
	Eigen::MatrixXd _C;  // Measurement matrix
	Eigen::MatrixXd _Q;  // Process noise covariance
	Eigen::MatrixXd _R;  // Measurement noise covariance
	Eigen::MatrixXd _P;  // Estimate error covariance
	Eigen::MatrixXd _K;  // Kalman gain
	/**
	 * 初始化的估计误差协方差
	 */
	Eigen::MatrixXd P0; // Initialization of estimate error covariance
	int _m = 0;          // Number of measurements
	int _n = 0;          // Number of states
	int _k = 0;          // Number of controls
	bool initialized = false;
	// n-size identity
	Eigen::MatrixXd I;
	Eigen::VectorXd _x_hat;            /// Estimated states 估计的状态
	Eigen::VectorXd _x_hat_predicted;  /// Updated state  重新的更新的状态
};

} // end of aura
} // end of aura_asd

