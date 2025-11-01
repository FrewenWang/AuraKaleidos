
/**
 * @file kalman_filter.cpp
 **/
#include "KalmanFilter.h"

#include <utility>
#include <iostream>

namespace aura {
namespace aura_asd {

KalmanFilter::KalmanFilter(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
						   Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd P)
		: _A(std::move(A)), _B(std::move(B)), _C(std::move(C)), _Q(std::move(Q)), _R(std::move(R)), P0(std::move(P)),
		  _m(_C.rows()), _n(_A.rows()), _k(_B.rows()), initialized(false),
		  I(_n, _n), _x_hat(_n), _x_hat_predicted(_n) {
	I.setIdentity();
}

void KalmanFilter::init() {
	/// 估计的状态
	_x_hat.setZero();
	/// 重新更新的状态设置为初始化的状态
	_x_hat_predicted.setZero();
	/// 初始化的估计误差协方差
	_P = P0;
	/// 初始化状态设置为true
	initialized = true;
}

void KalmanFilter::init(const Eigen::VectorXd &x0) {
	/// 估计的状态的设置为初始化的状态
	_x_hat = x0;
	/// 重新更新的状态设置为初始化的状态
	_x_hat_predicted = x0;
	/// 初始化的估计误差协方差
	_P = P0;
	initialized = true;
}

void KalmanFilter::predict(const Eigen::VectorXd &u) {
	if (!initialized) {
		std::cout << "Filter is not initialized! Initializing with trivial state.";
		init();
	}
	/// 更新的系统状态 = 系统动力学矩阵 * 当前系统估计状态 + 系统控制矩阵 * 输入控制
	_x_hat_predicted = _A * _x_hat + _B * u;
	_P = _A * _P * _A.transpose() + _Q;
}

void KalmanFilter::predict() {
	if (!initialized) {
		std::cout << "Filter is not initialized! Initializing with trivial state.";
		init();
	}
	_x_hat_predicted = _A * _x_hat;
	_P = _A * _P * _A.transpose() + _Q;
}

void KalmanFilter::update(const Eigen::VectorXd &y) {
	if (!initialized) {
		std::cout << "Filter is not initialized! Initializing with trivial state.";
		init();
	}
	
	_K = _P * _C.transpose() * (_C * _P * _C.transpose() + _R).inverse();
	_x_hat_predicted += _K * (y - _C * _x_hat_predicted);
	_P = (I - _K * _C) * _P;
	_x_hat = _x_hat_predicted;
}

void KalmanFilter::setPredictedState(const Eigen::VectorXd &x_hat_new) {
	_x_hat_predicted = x_hat_new;
}

void KalmanFilter::setUpdatedState(const Eigen::VectorXd &x_hat) {
	_x_hat = x_hat;
}

const Eigen::VectorXd &KalmanFilter::predictedState() const {
	return _x_hat_predicted;
}

const Eigen::VectorXd &KalmanFilter::updatedState() const {
	return _x_hat;
}

void KalmanFilter::updateDynamics(const Eigen::MatrixXd A) {
	this->_A = A;
}

void KalmanFilter::updateOutput(const Eigen::MatrixXd C) {
	this->_C = C;
}

} // namespace aura
} // namespace aura_asd
