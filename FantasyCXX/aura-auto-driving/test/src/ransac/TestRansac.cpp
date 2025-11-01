//
// Created by Frewen.Wang on 2024/10/24.
//
#include <random>

#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aad/kalman_filter/KalmanCVFilter.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

const static char *TAG = "TestRansac";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;

// 三次曲线模型：y = a*x³ + b*x² + c*x + d
struct CubicCurve {
  double c3, c2, c1, c0;
};

/**
 * 使用最小二乘法来进行三次曲线的拟合
 * @param x
 * @param y
 * @return
 */
CubicCurve leastSquaresFit(const vector<double> &x, const vector<double> &y) {
  // 数据点数量
  int n = x.size();
  // 设计矩阵：n行×4列，对应4个参数，也就是说针对每个店都给出一个参数匹配？？
  MatrixXd A(n, 4);
  // 观测值向量，由于存放真实的Y值坐标
  VectorXd B(n);

  // 构建正规方程的系数矩阵A和观测向量B
  for (int i = 0; i < n; ++i) {
    // 当前点的x值
    double xi = x[i];
    // 填充矩阵A的每一行（对应一个数据点的多项式展开）
    A(i, 0) = xi * xi * xi;     // x³项
    A(i, 1) = xi * xi;          // x²项
    A(i, 2) = xi;               // x项
    A(i, 3) = 1;                // 常数项
    B(i) = y[i];                        // 当前点的y值
  }

  // 解正规方程 (A^T A) * params = A^T B
  // 使用LDLT分解提高计算稳定性（相比直接求逆）
  // 返回拟合结果的结构体
  Vector4d params = (A.transpose() * A).ldlt().solve(A.transpose() * B);

  return {params[0], params[1], params[2], params[3]};
}

/**
 * 
 * @param x
 * @param y 
 * @param max_iterations    // 最大迭代次数 max_iterations = 100
 * @param threshold         // 内点判定阈值（误差小于此值为内点） threshold = 0.1
 * @param min_samples       // 最小样本数（三次曲线需要4个点解方程）
 * @return 
 */
CubicCurve ransacFit(const vector<double> &x, const vector<double> &y,
                     int max_iterations = 100, double threshold = 0.1,
                     int min_samples = 4) {
  // 随机数种子
  random_device rd;
  // 随机数生成器
  mt19937 gen(rd());
  // 均匀分布用于随机采样
  uniform_int_distribution<> dis(0, x.size() - 1);
  // 保存最佳模型
  CubicCurve best_model;
  // 最佳模型对应的内点数
  int best_inliers = 0;
  // RANSAC主循环
  for (int iter = 0; iter < max_iterations; ++iter) {
    // 随机选择min_samples个样本
    vector<double> sample_x, sample_y;
    for (int i = 0; i < min_samples; ++i) {
      // 生成随机索引
      int idx = dis(gen);
      sample_x.push_back(x[idx]);
      sample_y.push_back(y[idx]);
    }

    // 用样本拟合临时模型
    CubicCurve model = leastSquaresFit(sample_x, sample_y);

    // 统计内点数量
    int inliers = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      double error = abs(model.c3 * pow(x[i], 3) +
                         model.c2 * pow(x[i], 2) +
                         model.c1 * x[i] +
                         model.c0 - y[i]);
      if (error < threshold) {
        ++inliers;
      }
    }

    // 更新最佳模型
    if (inliers > best_inliers) {
      best_inliers = inliers;
      best_model = model;
    }
  }

  // 用所有内点重新拟合最终模型
  vector<double> inlier_x, inlier_y;
  for (size_t i = 0; i < x.size(); ++i) {
    double error = abs(best_model.c3 * pow(x[i], 3) +
                       best_model.c2 * pow(x[i], 2) +
                       best_model.c1 * x[i] +
                       best_model.c0 - y[i]);
    if (error < threshold) {
      inlier_x.push_back(x[i]);
      inlier_y.push_back(y[i]);
    }
  }

  return leastSquaresFit(inlier_x, inlier_y);
}

// 生成测试数据（带噪声和异常值）
void generateTestData(vector<double> &x, vector<double> &y) {
  mt19937 gen(123);
  normal_distribution<> d(0, 0.1); // 高斯噪声

  // 真实参数：y = 0.5x³ - 2x² + 1x + 3
  for (double xi = -3; xi <= 5; xi += 0.5) {
    x.push_back(xi);
    double noise = d(gen);

    // 添加10%的异常值
    if (rand() % 10 == 0) noise += 5.0 * (rand() % 2 ? 1 : -1);

    y.push_back(0.5 * pow(xi, 3) - 2 * pow(xi, 2) + xi + 3 + noise);
  }
}

class TestRansac : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGE(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGE(TAG, "TearDownTestSuite");
  }
};


TEST_F(TestRansac, TestRansacDemo) {
  // 生成测试数据
  vector<double> x, y;
  generateTestData(x, y);

  // 最小二乘法拟合（直接使用全部数据，受异常值影响大）
  CubicCurve ls_model = leastSquaresFit(x, y);
  cout << "Least Squares Fit:\n"
       << "c3 = " << ls_model.c3 << "\n"
       << "c2 = " << ls_model.c2 << "\n"
       << "c1 = " << ls_model.c1 << "\n"
       << "c0 = " << ls_model.c0 << endl;

  // RANSAC拟合（鲁棒性强，可排除异常值）
  // 参数：最大迭代100次，内点阈值0.5，最小样本4个
  CubicCurve ransac_model = ransacFit(x, y, 100, 0.5);
  cout << "\nRANSAC Fit:\n"
       << "c3 = " << ransac_model.c3 << "\n"
       << "c2 = " << ransac_model.c2 << "\n"
       << "c1 = " << ransac_model.c1 << "\n"
       << "c0 = " << ransac_model.c0 << endl;

}
