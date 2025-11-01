//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "../kalman_filter/BasicKalmanFilter.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

const static char *TAG = "TestBasicKalmanFilter";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;

class TestBasicKalmanFilter : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGE(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGE(TAG, "TearDownTestSuite");
	}
};


TEST_F(TestBasicKalmanFilter, testBasicKalmanCVFilterInit) {
	ALOGD(TAG, "============== testBasicKalmanCVFilterInit ==============");
	int n = 3; // Number of states  我们所要进行滤波处理的状态量 TODO 为什么是三个状态量。
	int m = 1; // Number of measurements  观测的数量，我们进行一个一个的观测结果进行滤波
	int c = 1; // Number of control inputs	控制的输入数量
	
	/// 声明我们各个kalman filter的各个状态和观测矩阵、协方差矩阵
	Eigen::MatrixXd A(n, n); // System dynamics matrix
	Eigen::MatrixXd B(n, c); // Input control matrix
	Eigen::MatrixXd C(m, n); // Output matrix
	Eigen::MatrixXd Q(n, n); // Process noise covariance
	Eigen::MatrixXd R(m, m); // Measurement noise covariance
	Eigen::MatrixXd P(n, n); // Estimate error covariance
	
	double dt = 1.0 / 30;
	// Discrete LTI projectile motion, measuring position only
	A << 1, dt, 0, 0, 1, dt, 0, 0, 1;
	B << 0, 0, 0;
	C << 1, 0, 0;
	
	// Reasonable covariance matrices   合理的协方差矩阵
	Q << .05, .05, .0, .05, .05, .0, .0, .0, .0;
	R << 5;
	P << .1, .1, .1, .1, 10000, 10, .1, 10, 100;
	
	std::cout << "=================A:系统动力学矩阵====================== \n" << A << std::endl;
	std::cout << "=================B:输入控制矩阵====================== \n" << B << std::endl;
	std::cout << "=================C:输出矩阵====================== \n" << C << std::endl;
	std::cout << "=================Q:过程噪声协方差====================== \n" << Q << std::endl;
	std::cout << "=================R:观测早上协方差====================== \n" << R << std::endl;
	std::cout << "=================P:估计误差协方差====================== \n" << P << std::endl;
	
	// Construct the filter
	BasicKalmanFilter kf(A, B, C, Q, R, P);
	
	// List of noisy position measurements (y)
	std::vector<double> measurements = {
			1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
			1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
			2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
			2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
			2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
			2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
			2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
			1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
			0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
	};
	
	// Best guess of initial states
	Eigen::VectorXd x0(n);
	// 初始化动力学
	x0 << measurements[0], 0, -9.81;
	kf.init(x0);
	
	// Feed measurements into filter, output estimated states
	double t = 0;
	Eigen::VectorXd y(m), u(c);
	std::cout << "========================开始进行卡尔曼滤波处理============================" << std::endl;
	std::cout << "t = " << t << ", " << "x_hat[0]: " << kf.state().transpose() << std::endl;
	for (int i = 0; i < measurements.size(); i++) {
		t += dt;
		y << measurements[i];
		u << 0; // example with zero control input
		kf.predict(u);
		kf.update(y);
		std::cout << "t = " << t << ", " << "y[" << i << "] = " << y.transpose()
				  << ", x_hat[" << i << "] = " << kf.state().transpose() << std::endl;
	}
	
	
}