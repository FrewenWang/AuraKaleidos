//
// Created by Frewen.Wang on 2024/10/23.
//

#include "aura/aad/kalman_filter/KalmanCVFilter.h"
#include "aura/aura_utils/utils/AuraLog.h"

namespace aura {
namespace aad {

KalmanCVFilter::KalmanCVFilter() {
	name_ = "KalmanCVFilter";
}

KalmanCVFilter::~KalmanCVFilter() = default;

bool KalmanCVFilter::Init(const Eigen::VectorXd &initial_belief_states, const Eigen::MatrixXd &initial_uncertainty) {
	if (initial_uncertainty.rows() != initial_uncertainty.cols()) {
		ALOGE(name_.c_str(), "the cols and rows of uncertainty matrix should be equal");
		return false;
	}
	return true;
}

bool KalmanCVFilter::predict(const double &delta_t) {
	return false;
}

bool KalmanCVFilter::update(const Eigen::VectorXd &cur_observation, const Eigen::MatrixXd &cur_observation_uncertainty,
							const bool motion_constraint_valid_flag) {
	if (!init_) {
		ALOGE(name_.c_str(), "update: Kalman Filter initialize not successfully");
		return false;
	}
	
	
	return true;
}

Eigen::VectorXd KalmanCVFilter::getStates() const {
	return Eigen::VectorXd();
}


} // aura_aad
} // aura