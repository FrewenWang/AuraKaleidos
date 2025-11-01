#pragma once

#include "Eigen/Dense"
#include <string>

namespace aura {
namespace aad {
/**
 * @brief base filter inference
 */
class BaseKalmanFilter {
public:
	/**
	 * 观测传感器类型
	 */
	enum class SensorType {
		Radar = 0, Lidar, Camera, Unknown
	};
	
	/**
	 * constructor
	 */
	BaseKalmanFilter() : states_num(0), init_(false), name_("BaseKalmanFilter") { }
	
	/**
	 * destructor
	 */
	virtual ~BaseKalmanFilter() = default;
	
	/**
	 * 卡尔曼滤波器初始化
	 * @param global_states a vector contains system states(position, velocity etc.)
	 * @param global_uncertainty  a covariance matrix which indicate the uncertainty of each system state
	 * @return
	 */
	virtual bool Init(const Eigen::VectorXd &global_states, const Eigen::MatrixXd &global_uncertainty) = 0;
	
	// @brief predict the current state and uncertainty of system
	// @params[IN] transform_matrix: transform the state from the
	//             pre moment to current moment
	// @params[IN] env_uncertainty_matrix: the uncertainty brought by
	//             the environment when predict.
	
	// #ifdef CA
	//     virtual bool Predict(
	//             const Eigen::MatrixXd &transform_matrix,
	//             const Eigen::MatrixXd &env_uncertainty_matrix) = 0;
	// #else
	//     virtual bool Predict(const double &delta_t) = 0;
	// #endif
	
	// @brief use the current observation to correct the predict
	// @params[IN] cur_observation: the observation in current time
	// @params[IN] cur_observation_uncertainty: the uncertainty of
	//             the observation
	// virtual bool Correct(
	//         const Eigen::VectorXd &cur_observation,
	//         const Eigen::MatrixXd &cur_observation_uncertainty) = 0;
	
	// @brief set the control matrix
	// virtual bool SetControlMatrix(const Eigen::MatrixXd &c_matrix) = 0;
	
	/**
	 * @brief get the system states
	 * 获取系统状态
	 * @return
	 */
	virtual Eigen::VectorXd getStates() const = 0;
	
	/**
	 * @brief get the name of the filter
	 * @return
	 */
	std::string Name() {
		return name_;
	}
	
	/**
	 * 获取系统状态的数量
	 * @return
	 */
	int getStateNum() const {
		return states_num;
	}
	
	/**
	 * 获取全局状态
	 * @return
	 */
	Eigen::VectorXd GetStates() const {
		return global_states_;
	}
	
	/**
	 * 获取不确定观测矩阵
	 * @return
	 */
	Eigen::MatrixXd GetUncertainty() const {
		return global_uncertainty_;
	}

protected:
	/**
	 * @brief the number of the system states
	 */
	int states_num;
	
	// @brief whether the filter has been init
	bool init_;
	
	///  @brief 卡尔曼滤波器的名字
	std::string name_;
	
	Eigen::MatrixXd transform_matrix_;
	Eigen::VectorXd global_states_;
	Eigen::VectorXd innovation_;
	Eigen::MatrixXd global_uncertainty_;
	Eigen::MatrixXd env_uncertainty_;
	/**
	 * 当前观测结果数据
	 */
	Eigen::MatrixXd cur_observation_;
	Eigen::MatrixXd cur_observation_uncertainty_;
	Eigen::MatrixXd c_matrix_;
};

}
}
