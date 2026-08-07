//
// Created by Frewen.Wang on 2024/10/23.
//

#include "aura/aad/kalman_filter/KalmanCVFilter.h"
#include "aura/aura_utils/utils/AuraLog.h"

namespace aura {
namespace aad {

KalmanCVFilter::KalmanCVFilter() {
		name_ = "KalmanCVFilter";
		states_num = 4;
		global_states_ = Eigen::VectorXd::Zero(states_num);
		global_uncertainty_ = Eigen::MatrixXd::Identity(states_num, states_num);
		transform_matrix_ = Eigen::MatrixXd::Identity(states_num, states_num);
		env_uncertainty_ = Eigen::MatrixXd::Identity(states_num, states_num) * 1e-3;
}

KalmanCVFilter::~KalmanCVFilter() = default;

bool KalmanCVFilter::Init(const Eigen::VectorXd &initial_belief_states, const Eigen::MatrixXd &initial_uncertainty) {
		if (initial_uncertainty.rows() != initial_uncertainty.cols()) {
		ALOGE(name_.c_str(), "the cols and rows of uncertainty matrix should be equal");
			return false;
		}
		if (initial_belief_states.size() != states_num ||
			initial_uncertainty.rows() != states_num ||
			!initial_belief_states.allFinite() || !initial_uncertainty.allFinite()) {
			ALOGE(name_.c_str(), "invalid initial state or uncertainty dimensions");
			return false;
		}
		global_states_ = initial_belief_states;
		global_uncertainty_ = initial_uncertainty;
		init_ = true;
		return true;
}

bool KalmanCVFilter::predict(const double &delta_t) {
		if (!init_ || delta_t < 0.0) {
			return false;
		}
		transform_matrix_.setIdentity();
		transform_matrix_(0, 2) = delta_t;
		transform_matrix_(1, 3) = delta_t;
		global_states_ = transform_matrix_ * global_states_;
		global_uncertainty_ = transform_matrix_ * global_uncertainty_ * transform_matrix_.transpose() +
				env_uncertainty_;
		return true;
}

bool KalmanCVFilter::update(const Eigen::VectorXd &cur_observation, const Eigen::MatrixXd &cur_observation_uncertainty,
							const bool motion_constraint_valid_flag) {
	if (!init_) {
		ALOGE(name_.c_str(), "update: Kalman Filter initialize not successfully");
		return false;
	}
		const auto observation_size = cur_observation.size();
		if (observation_size <= 0 || observation_size > states_num ||
			cur_observation_uncertainty.rows() != observation_size ||
			cur_observation_uncertainty.cols() != observation_size) {
			return false;
		}
		Eigen::MatrixXd observation_matrix = Eigen::MatrixXd::Zero(observation_size, states_num);
		observation_matrix.block(0, 0, observation_size, observation_size).setIdentity();
		innovation_ = cur_observation - observation_matrix * global_states_;
		Eigen::MatrixXd residual_covariance = observation_matrix * global_uncertainty_ *
				observation_matrix.transpose() + cur_observation_uncertainty;
		Eigen::LDLT<Eigen::MatrixXd> solver(residual_covariance);
		if (solver.info() != Eigen::Success) {
			return false;
		}
		Eigen::MatrixXd gain = solver.solve(observation_matrix * global_uncertainty_).transpose();
		global_states_ += gain * innovation_;
		Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(states_num, states_num);
		Eigen::MatrixXd correction = identity - gain * observation_matrix;
		global_uncertainty_ = correction * global_uncertainty_ * correction.transpose() +
				gain * cur_observation_uncertainty * gain.transpose();
		return true;
}

Eigen::VectorXd KalmanCVFilter::getStates() const {
		return global_states_;
}


} // aura_aad
} // aura
