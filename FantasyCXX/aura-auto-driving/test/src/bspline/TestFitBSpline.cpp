//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aura_utils/utils/FileUtil.h"
#include "aura/aura_utils/utils/StringUtil.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

// #include "matplotlibcpp.h"
const static char *TAG = "TestFitBSpline";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;
using namespace aura::utils;

/**
 * 使用boost进行点集的平滑
 * @param xs
 * @param ys
 * @param smoothRatio
 * @return
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> smoothPointsToCurveBySplrep(
		const Eigen::VectorXd &xs,
		const Eigen::VectorXd &ys,
		double smoothRatio = 0.1) {
	
	// Create a vector of y values from Eigen::VectorXd
	std::vector<double> y_values(ys.data(), ys.data() + ys.size());
	
	// Define the step size (uniform spacing assumed)
	double step = xs[1] - xs[0];
	
	// Create the cubic B-spline interpolator
	auto spline = boost::math::interpolators::cardinal_cubic_b_spline<double>(
			y_values.begin(), y_values.end(), xs[0], step, smoothRatio);
	
	// Define the start and end of the new x range
	int start = static_cast<int>(xs[0]);
	int end = static_cast<int>(xs[xs.size() - 1]);
	
	// Generate new x values
	Eigen::VectorXd newx = Eigen::VectorXd::LinSpaced(end - start + 1, start, end);
	
	// Create new y values based on the spline interpolator
	Eigen::VectorXd newy(newx.size());
	for (int i = 0; i < newx.size(); ++i) {
		newy[i] = spline(newx[i]);
	}
	
	return std::make_pair(newx, newy);
}

// B样条基函数的递归计算
double N(int i, int k, double t, const vector<double> &knots) {
	if (k == 1) {
		return (t >= knots[i] && t < knots[i + 1]) ? 1.0 : 0.0;
	} else {
		double coef1 = (t - knots[i]) / (knots[i + k - 1] - knots[i]);
		double coef2 = (knots[i + k] - t) / (knots[i + k] - knots[i + 1]);
		return coef1 * N(i, k - 1, t, knots) + coef2 * N(i + 1, k - 1, t, knots);
	}
}

// 计算控制点，通过最小二乘法和正则化来拟合
vector<Vector2d> fitBSpline(const vector<Vector2d> &points, int degree, double smooth_factor) {
	int n = points.size();
	int m = n + degree + 1;
	
	// 均匀分布的节点向量
	vector<double> knots(m);
	for (int i = 0; i < m; ++i) {
		knots[i] = i / double(m - 1);
	}
	
	// 基函数矩阵
	MatrixXd basisMatrix(n, n);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			basisMatrix(i, j) = N(j, degree + 1, i / double(n - 1), knots);
		}
	}
	
	// 正则化矩阵
	MatrixXd regularization = smooth_factor * MatrixXd::Identity(n, n);
	
	// 目标矩阵
	MatrixXd targetMatrix(n, 2);
	for (int i = 0; i < n; ++i) {
		targetMatrix(i, 0) = points[i](0);
		targetMatrix(i, 1) = points[i](1);
	}
	
	// 通过最小二乘法求解控制点
	MatrixXd controlPoints = (basisMatrix.transpose() * basisMatrix + regularization).ldlt().solve(
			basisMatrix.transpose() * targetMatrix);
	
	// 将结果转换为Vector2d
	vector<Vector2d> result;
	for (int i = 0; i < n; ++i) {
		result.emplace_back(controlPoints(i, 0), controlPoints(i, 1));
	}
	
	return result;
}

// 绘制B样条曲线
void plotBSpline(const vector<Vector2d> &controlPoints, int degree, const vector<double> &knots) {
	vector<double> x, y;
	
	// // 生成曲线点
	// for (double t = knots[degree]; t < knots[knots.size() - degree - 1]; t += 0.01) {
	// 	Vector2d point(0.0, 0.0);
	// 	for (size_t i = 0; i < controlPoints.size(); ++i) {
	// 		double basis = N(i, degree + 1, t, knots);
	// 		point += basis * controlPoints[i];
	// 	}
	// 	x.push_back(point.x());
	// 	y.push_back(point.y());
	// }
	//
	// // 绘制
	// plt::plot(x, y);
	// plt::scatter(x, y, 10.0);
	// plt::title("B-Spline Curve");
	// plt::show();
}


vector<std::string> results;

class TestFitBSpline : public testing::Test {
public:
	
	static void SetUpTestSuite() {
		ALOGE(TAG, "SetUpTestSuite");
		//// 读取文件中的的坐标带你
		string path = "/home/baiduiov/03.ProgramSpace/01"
					  ".WorkSpace/AuraKaleidoScope/AuraKaleidoCXX/aura-auto-driving/test/src/bspline/points.txt";
		std::string content;
		aura::utils::FileUtil::readFile(path, content);
		// ALOGD(TAG, content.c_str());
		StringUtil::splitStr(content, ',', results);
		// 打印原始点的逻辑
		std::stringstream ss;
		ss << "OriginPoints:";
		for (auto &point: results) {
			ss << point.c_str() << ",";
		}
		ALOGD(TAG, ss.str().c_str());
	}
	
	static void TearDownTestSuite() {
		ALOGE(TAG, "TearDownTestSuite");
	}
};


TEST_F(TestFitBSpline, testFitBSplineEigen) {
	/// 将分割好的所有坐标
	vector<Vector2d> points;
	for (int i = 0; i < results.size() - 1; i += 2) {
		double x = std::stod(results[i]);
		double y = std::stod(results[i + 1]);
		points.emplace_back(x, y);
	}
	
	/// 证明：gpt给的结果不可行
	int degree = 3; // B样条的度数
	double smooth_factor = 0.1; // 平滑因子
	//拟合B样条
	vector<Vector2d> outputPoints = fitBSpline(points, degree, smooth_factor);
	// 打印优化之后的点
	std::stringstream ss_output;
	ss_output << "OutputPoints:";
	for (auto &point: outputPoints) {
		ss_output << point(0) << "," << point(1) << ",";
	}
	ALOGD(TAG, ss_output.str().c_str());
}

TEST_F(TestFitBSpline, testFitBSplineBoost) {
	/// 将分割好的所有坐标
	vector<Vector2d> points;
	for (int i = 0; i < results.size() - 1; i += 2) {
		double x = std::stod(results[i]);
		double y = std::stod(results[i + 1]);
		points.emplace_back(x, y);
	}
	
	/// 证明：gpt给的结果不可行
	int degree = 3; // B样条的度数
	double smooth_factor = 0.1; // 平滑因子
	Eigen::VectorXd xs;
	Eigen::VectorXd ys;
	// std::pair<Eigen::VectorXd, Eigen::VectorXd>  outputPoints = smoothPointsToCurveBySplre(xs,ys);
	
	// 打印优化之后的点
	// std::stringstream ss_output;
	// ss_output << "OutputPoints:";
	// for (auto &point: outputPoints) {
	// 	ss_output << point(0) << "," << point(1) << ",";
	// }
	// ALOGD(TAG, ss_output.str().c_str());
}


TEST_F(TestFitBSpline, testFitBSplineDraw) {
	// // 样例点集（超过100个点）
	// vector<Vector2d> points;
	// for (int i = 0; i <= 100; ++i) {
	// 	double x = i / 10.0;
	// 	double y = sin(x) + (rand() % 10) / 10.0; // 添加一些波动
	// 	points.emplace_back(x, y);
	// }
	//
	// int degree = 3; // B样条的度数
	// double smooth_factor = 0.1; // 平滑因子
	//
	// // 拟合B样条
	// vector<Vector2d> controlPoints = fitBSpline(points, degree, smooth_factor);
	//
	// // 均匀分布的节点向量
	// int n = points.size();
	// vector<double> knots(n + degree + 1);
	// for (int i = 0; i < knots.size(); ++i) {
	// 	knots[i] = i / double(knots.size() - 1);
	// }
	//
	// // 绘制B样条
	// plotBSpline(controlPoints, degree, knots);
	
}