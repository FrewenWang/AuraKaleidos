
#include "SquareRootKalmanFilter.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <sstream>

namespace aura {
namespace aura_asd {

void SRKalmanFilter::Init() {
	sr_covariance_matrix_.setZero(covariance_matrix_.rows(), covariance_matrix_.cols());
	for (auto i = 0; i < sr_covariance_matrix_.rows(); ++i) {
		sr_covariance_matrix_(i, i) = std::sqrt(covariance_matrix_(i, i));
	}
}

void SRKalmanFilter::Predict() {
	// 1.predict state, Xk = Fk * Xk + Bk * uk
	state_vector_ = state_trans_matrix_ * state_vector_ + control_matrix_ * control_vector_;  // 4*1
	// 2.compute covariance, Pk = Fk * Pk * Fk^T + Tk * Qk * Tk^T
	sr_process_noise_covariance_matrix_.setZero(  // 3*3, diagonal matrix
			process_noise_covariance_matrix_.rows(), process_noise_covariance_matrix_.cols());
	for (auto i = 0; i < sr_process_noise_covariance_matrix_.rows(); ++i) {
		sr_process_noise_covariance_matrix_(i, i) = std::sqrt(process_noise_covariance_matrix_(i, i));
	}
	// (Pk^1/2)^T * Fk^T
	Eigen::MatrixXd sr_pk_fk = sr_covariance_matrix_.transpose() * state_trans_matrix_.transpose();  // 4*4
	// (Qk^1/2)^T * Tk^T
	Eigen::MatrixXd sr_tk_qk =
			sr_process_noise_covariance_matrix_.transpose() * process_noise_model_matrix_.transpose();  // 3*4
	// 2.1 form A = ((Pk^1/2)^T * Fk^T : (Qk^1/2)^T * Tk^T), QR factorization
	Eigen::MatrixXd sr_pk_fk_tk_qk = Eigen::MatrixXd::Zero(sr_pk_fk.rows() + sr_tk_qk.rows(), sr_pk_fk.cols());  // 7*4
	sr_pk_fk_tk_qk << sr_pk_fk, sr_tk_qk;
	Eigen::HouseholderQR<Eigen::MatrixXd> qr;
	qr.compute(sr_pk_fk_tk_qk);
	Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();
	Eigen::MatrixXd q = qr.householderQ();
	r.conservativeResize(sr_covariance_matrix_.rows(), sr_covariance_matrix_.cols());
	// 2.2 (Pk^1/2)^T = r, Pk^1/2 is lower triangular matrix
	sr_covariance_matrix_ = r.transpose();  // 4*4
	covariance_matrix_ = sr_covariance_matrix_ * sr_covariance_matrix_.transpose();
	process_noise_matrix_ =  // 4*4
			process_noise_model_matrix_ * sr_process_noise_covariance_matrix_ *
			sr_process_noise_covariance_matrix_.transpose() * process_noise_model_matrix_.transpose();
	
}

void SRKalmanFilter::Correct(const Eigen::VectorXd &measurement_vector) {
	// ak = (Pk^1/2)^T * Hk^T
	Eigen::MatrixXd ak = sr_covariance_matrix_.transpose() * measurement_trans_matrix_.transpose();  // 4*n
	Eigen::MatrixXd ak_2 = ak.transpose() * ak;
	Eigen::MatrixXd ak_measure_noise = ak_2 + measurement_noise_matrix_;
	// bk = (ak^T * ak + Rk)^-1
	Eigen::MatrixXd bk = ak_measure_noise.inverse();         // n*n
	Eigen::MatrixXd bk_rk = bk * measurement_noise_matrix_;  // n*n
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	ldlt.compute(bk_rk);
	Eigen::MatrixXd l = ldlt.matrixL();
	Eigen::VectorXd d = ldlt.vectorD();
	Eigen::MatrixXd sqrt_d = Eigen::MatrixXd::Zero(bk_rk.rows(), bk_rk.cols());
	for (auto i = 0; i < d.rows(); ++i) {
		sqrt_d(i, i) = std::sqrt(d(i));
	}
	Eigen::MatrixXd sqrt_bk_rk = l * sqrt_d;
	Eigen::MatrixXd bk_rk_i = Eigen::MatrixXd::Identity(bk_rk.rows(), bk_rk.cols());
	bk_rk_i += sqrt_bk_rk;
	// rk = (I + (bk * Rk)^1/2)^-1
	Eigen::MatrixXd rk = bk_rk_i.inverse();  // n*n
	
	for (auto i = 0; i < rk.rows(); ++i) {
		for (auto j = 0; j < rk.cols(); ++j) {
			if (std::isnan(rk(i, j))) {
				rk = Eigen::MatrixXd::Identity(bk_rk.rows(), bk_rk.cols());
				break;
			}
		}
	}
	kalman_gain_matrix_ = sr_covariance_matrix_ * ak * bk;  // 4*n
	residual_vector_ = measurement_vector - measurement_trans_matrix_ * state_vector_;
	estimate_measurement_vector_ = measurement_trans_matrix_ * state_vector_;
	// xk = xk + K * (z - Hk * xk)
	Eigen::VectorXd delta_state_vector = kalman_gain_matrix_ * residual_vector_;
	if (particleFeedBack) {
		SetFeedBackRatio(delta_state_vector);
	}
	// if (false) {
	//   SetFeedBackRatio(delta_state_vector);
	// }
	// ADEBUG << "particel feed back: " << partical_feed_back_;
	state_vector_ += delta_state_vector;  // 4*1
	// Pk^1/2 -= K * rk * ak^T
	sr_covariance_matrix_ -= kalman_gain_matrix_ * rk * ak.transpose();  // 4*4
	covariance_matrix_ = sr_covariance_matrix_ * sr_covariance_matrix_.transpose();
	measurement_vector_ = measurement_vector;
	
}

void SRKalmanFilter::SymmetricCorrect(const Eigen::VectorXd &measurement_vector) {
	Eigen::MatrixXd ak = sr_covariance_matrix_.transpose() * measurement_trans_matrix_.transpose();  // 4*n
	Eigen::MatrixXd ak_2 = ak.transpose() * ak;
	Eigen::MatrixXd ak_measure_noise = ak_2 + measurement_noise_matrix_;
	// bk = (ak^T * ak + Rk)^-1
	Eigen::MatrixXd bk = ak_measure_noise.inverse();        // n*n
	kalman_gain_matrix_ = sr_covariance_matrix_ * ak * bk;  // 4*n
	residual_vector_ = measurement_vector - measurement_trans_matrix_ * state_vector_;
	// xk = xk + K * (z - Hk * xk)
	state_vector_ += kalman_gain_matrix_ * residual_vector_;  // 4*1
	
	// get sr_covariance with Joseph form covariance
	Eigen::MatrixXd cov_I = Eigen::MatrixXd::Identity(covariance_matrix_.rows(), covariance_matrix_.cols());
	Eigen::MatrixXd cov_a =
			sr_covariance_matrix_.transpose() * (cov_I - kalman_gain_matrix_ * measurement_trans_matrix_).transpose();
	Eigen::MatrixXd sr_r = Eigen::MatrixXd::Zero(measurement_noise_matrix_.rows(), measurement_noise_matrix_.cols());
	for (auto i = 0; i < measurement_noise_matrix_.rows(); ++i) {
		sr_r(i, i) = std::sqrt(measurement_noise_matrix_(i, i));
	}
	Eigen::MatrixXd cov_b = sr_r.transpose() * kalman_gain_matrix_.transpose();
	Eigen::MatrixXd cov_a_b = Eigen::MatrixXd::Zero(cov_a.rows() + cov_b.rows(), cov_a.cols());  // 8*4
	cov_a_b << cov_a, cov_b;
	Eigen::HouseholderQR<Eigen::MatrixXd> qr;
	qr.compute(cov_a_b);
	Eigen::MatrixXd r = qr.matrixQR().triangularView<Eigen::Upper>();
	Eigen::MatrixXd q = qr.householderQ();
	r.conservativeResize(sr_covariance_matrix_.rows(), sr_covariance_matrix_.cols());
	for (auto i = 0; i < r.rows(); ++i) {
		for (auto j = 0; j < r.cols(); ++j) {
			if (std::isnan(r(i, j))) {
				r = sr_covariance_matrix_.transpose();
				break;
			}
		}
	}
	sr_covariance_matrix_ = r.transpose();  // 4*4
	
	covariance_matrix_ = sr_covariance_matrix_ * sr_covariance_matrix_.transpose();
	measurement_vector_ = measurement_vector;
}

std::string SRKalmanFilter::ToString() const {
	std::ostringstream oss;
	Eigen::VectorXd delta_state_vector = kalman_gain_matrix_ * residual_vector_;
	oss << "state_vector: rows=" << state_vector_.rows() << ", cols=" << state_vector_.cols()
		<< ", size=" << state_vector_.size() << ";\n"
		<< state_vector_ << "\n"
		<< "state_trans_matrix: rows=" << state_trans_matrix_.rows() << ", cols=" << state_trans_matrix_.cols()
		<< ", size=" << state_trans_matrix_.size() << ";\n"
		<< state_trans_matrix_ << "\n"
		<< "control_matrix: rows=" << control_matrix_.rows() << ", cols=" << control_matrix_.cols()
		<< ", size=" << control_matrix_.size() << ";\n"
		<< control_matrix_ << "\n"
		<< "control_vector: rows=" << control_vector_.rows() << ", cols=" << control_vector_.cols()
		<< ", size=" << control_vector_.size() << ";\n"
		<< control_vector_ << "\n"
		<< "sr_covariance_matrix: rows=" << sr_covariance_matrix_.rows() << ", cols=" << sr_covariance_matrix_.cols()
		<< ", size=" << sr_covariance_matrix_.size() << ";\n"
		<< sr_covariance_matrix_ << "\n"
		<< "covariance_matrix: rows=" << covariance_matrix_.rows() << ", cols=" << covariance_matrix_.cols()
		<< ", size=" << covariance_matrix_.size() << ";\n"
		<< covariance_matrix_ << "\n"
		<< "sr_process_noise_covariance_matrix: rows=" << sr_process_noise_covariance_matrix_.rows()
		<< ", cols=" << sr_process_noise_covariance_matrix_.cols()
		<< ", size=" << sr_process_noise_covariance_matrix_.size() << ";\n"
		<< sr_process_noise_covariance_matrix_ << "\n"
		<< "process_noise_matrix: rows=" << process_noise_matrix_.rows() << ", cols=" << process_noise_matrix_.cols()
		<< ", size=" << process_noise_matrix_.size() << ";\n"
		<< process_noise_matrix_ << "\n"
		<< "measurement_vector: rows=" << measurement_vector_.rows() << ", cols=" << measurement_vector_.cols()
		<< ", size=" << measurement_vector_.size() << ";\n"
		<< measurement_vector_ << "\n"
		<< "estimate_measurement_vector: rows=" << estimate_measurement_vector_.rows() << ", cols="
		<< estimate_measurement_vector_.cols()
		<< ", size=" << estimate_measurement_vector_.size() << ";\n"
		<< estimate_measurement_vector_ << "\n"
		<< "measurement_trans_matrix: rows=" << measurement_trans_matrix_.rows()
		<< ", cols=" << measurement_trans_matrix_.cols() << ", size=" << measurement_trans_matrix_.size() << ";\n"
		<< measurement_trans_matrix_ << "\n"
		<< "measurement_noise_matrix: rows=" << measurement_noise_matrix_.rows()
		<< ", cols=" << measurement_noise_matrix_.cols() << ", size=" << measurement_noise_matrix_.size() << ";\n"
		<< measurement_noise_matrix_ << "\n"
		<< "kalman_gain_matrix: rows=" << kalman_gain_matrix_.rows() << ", cols=" << kalman_gain_matrix_.cols()
		<< ", size=" << kalman_gain_matrix_.size() << ";\n"
		<< kalman_gain_matrix_ << "\n"
		<< "residual_vector: rows=" << residual_vector_.rows() << ", cols=" << residual_vector_.cols()
		<< ", size=" << residual_vector_.size() << ";\n"
		<< residual_vector_ << "\n"
		<< "delta_state_vector: rows=" << delta_state_vector.rows() << ", cols=" << delta_state_vector.cols()
		<< ", size=" << delta_state_vector.size() << ";\n"
		<< delta_state_vector << "\n";
	return oss.str();
}

void SRKalmanFilter::SetFeedBackRatio(Eigen::VectorXd &delta_vector) {
	if (feed_back_thresh_.size() != 4) {
		return;
	}
	if (delta_vector.size() != 4) {
		return;
	}
	// std::ostringstream oss;
	// oss << "before state vector: " << delta_vector << "\n";
	double feed_back_ratio = 1.0;
	for (int i = 0; i < 4; ++i) {
		if (fabs(delta_vector[i]) > feed_back_thresh_[i]) {
			double curr_ratio = feed_back_thresh_[i] / fabs(delta_vector[i]);
			feed_back_ratio = std::min(curr_ratio, feed_back_ratio);
		}
	}
	delta_vector *= feed_back_ratio;
	// delta_vector[0] *= feed_back_ratio;
	// delta_vector[1] *= feed_back_ratio;
	// oss << "after state vector: " << delta_vector.transpose() << "feed back ratio: " << feed_back_ratio << "\n";
	// ADEBUG << oss.str();
	return;
}

}  // namespace aura
}  // namespace aura_asd
