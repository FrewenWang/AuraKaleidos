#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "OsqpEigen/OsqpEigen.h"

using namespace std;
namespace plt = matplotlibcpp;

class SplineSmoother {
 public:
  SplineSmoother(const char* filename);
  ~SplineSmoother() = default;

 private:
  // step1: 整理输入的数据
  void GetSourceData(const char* filename);
  // step2: 构建目标函数的Q矩阵
  Eigen::SparseMatrix<double> CostFunction();
  // step3: 构建目标函数的A矩阵以及约束边界
  void ConstriantFunction();
  // step4: 求解问题
  bool Solution();
  // step5: 从参数方程转化为x,y路径点
  void ParametricToCartesian();

 public:
  int number;  //需要平滑的路径点的数量
  vector<pair<double, double>> input_points;  //原始路径点
  double heading_start;                       //起点的朝向
  double heading_end;                         //终点的朝向

  vector<pair<double, double>>
      anchor_points;  //限制路径的鞍点
                      //为什么要用鞍点?因为多项式曲线是参数方程，鞍点代表每段曲线的中间，t=0.5

  //这里matrix类对象需要用vector包一下，因为类对象单独作为成员函数，只能使用初始化列表初始化，非常受限制
  vector<Eigen::SparseMatrix<double>> Q;

  vector<Eigen::VectorXd> Derivative;
  vector<Eigen::VectorXd> bound;
  vector<Eigen::VectorXd> gradient;
  vector<Eigen::SparseMatrix<double>> A;
  //位置约束的范围
  double boundary = 0.2;
  int total_constriant_number;

  vector<Eigen::VectorXd> smooth_parametric;
  vector<pair<double, double>> smooth_points;  // result路径点
};