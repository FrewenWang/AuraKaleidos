//
// Created by Frewen.Wang on 2022/11/20.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <unistd.h>
#include <vector>
#include <random>

//  c++使用eigen提示"fatal error: Eigen/Dense: No such file or
//  directory"的解决办法
//  解决方法：https://blog.csdn.net/chengde6896383/article/details/88339643
//  解决方法一：
//  1. 将/usr/include/eigen3/Eigen递归复制到/usr/include/Eigen
//  cp -rf /usr/include/eigen3/Eigen /usr/include/Eigen -R
//  解决方法二：
//  #include <eigen3/Eigen/Dense>
#include <Eigen/Dense>       // Eigen 矩阵库

// 集成完成ceres之后，发现在ceres-solver代码里面提示：fatal error: 'glog/logging.h' file not found
// 但是我其实已经通过homebrew安装了glog：
// Warning: glog 0.6.0 is already installed and up-to-date.
// 所以怀疑homebrew安装的glog的路径，不能被系统默认识别，现在要看看homebrew帮我们把依赖库安装到哪里了。


// 错误二：
// integer_sequence_algorithm.h:75:31: error: no template named 'integer_sequence' in namespace 'std';
// did you mean '__integer_sequence'?
// `std::integer_sequence`是C++14引入的一个模板，用于生成整数序列，通常和`std::make_integer_sequence`、`std::index_sequence`一起用，用于编译时的序列生成，
// 比如展开参数包。那用户遇到的错误是说编译器找不到这个模板，反而提示了`__integer_sequence`，可能是某个编译器特定的内部实现。
// 由于我们目前使用的ceres2.2.0 应该是这个版本最低支持C++14
#include <ceres/ceres.h>     // Ceres 优化库

const static char *TAG = "TestCubicCurveFitting";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;

/**
 * 使用ceres-solver进行三次曲线的拟合
 * 以下是使用 Ceres Solver 实现三次曲线车道线拟合的完整 C++ 代码示例，包含详细注释和数据可视化。代码实现了带噪声数据的生成、最小二乘优化和结果可视化。
 */
class TestCubicCurveFitting : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGD(TAG, "SetUpTestSuite");
    }
    
    static void TearDownTestSuite() {
        ALOGD(TAG, "TearDownTestSuite");
    }
};

// ====================== 1. 曲线模型与残差定义 ====================== //
// 定义三次曲线残差计算器
struct CubicCurveResidual {
    CubicCurveResidual(double x, double y) : x_(x), y_(y) {}  // 构造函数保存观测点

    // 残差计算模板函数（自动微分）
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = a, params[1] = b, params[2] = c, params[3] = d
        // 计算 y = a*x³ + b*x² + c*x + d
        T predicted_y = params[0] * T(x_) * T(x_) * T(x_) +
                        params[1] * T(x_) * T(x_) +
                        params[2] * T(x_) +
                        params[3];
        // 残差 = 预测值 - 观测值
        // 所谓残差的计算： 就是我们拟合迭代的三次曲线的系数，带入观测X值得到的预测的Y值，和观测的Y值进行作差，得到的结果就是残差。
        residual[0] = predicted_y - T(y_);
        return true;
    }

private:
    const double x_;  // 观测点x坐标
    const double y_;  // 观测点y坐标
};

// ====================== 2. 数据生成函数 ====================== //
void generateData(vector<double>& x, vector<double>& y) {
    // 真实曲线参数：y = 0.5x³ - 2x² + x + 3
    // 我们看我们三次曲线拟合之后的数据，是不是能够和他比较相近
    const double a = 0.5, b = -2.0, c = 1.0, d = 3.0;
    // 加入一些高斯噪声
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise(0, 0.5);  // 高斯噪声

    // 生成-3到5之间的x值
    for (double xi = -3.0; xi <= 5.0; xi += 0.2) {
        double yi = a * xi*xi*xi + b * xi*xi + c * xi + d;
        // 添加噪声
        yi += noise(gen);

        // 添加10%异常值
        if (rand() % 10 == 0) {
            yi += 5.0 * (rand()%2 ? 1 : -1);
        }
        x.push_back(xi);
        y.push_back(yi);
    }
}


TEST_F(TestCubicCurveFitting, testCubicCurveFitting) {
    ALOGD(TAG, "============== testCubicCurveFitting ==============");
    // ---------- 步骤1：生成测试数据 ---------- //
    vector<double> x_data, y_data;
    // 这一步执行完毕之后，就是我们已经加了高斯噪声的观测结果，我们要基于这个结果拟合出三次曲线。
    generateData(x_data, y_data);

    // ---------- 步骤2：设置优化参数 ---------- //
    double params[4] = {0.0, 0.0, 0.0, 0.0};  // 初始参数 [a, b, c, d]

    // ---------- 步骤3：构建最小二乘问题 ---------- //
    ceres::Problem problem;
    for (size_t i = 0; i < x_data.size(); ++i) {
        // 对每个数据点添加残差块
        ceres::CostFunction* cost_function =
            // 残差维度1，所在残差维度是1，我们我们只计算residual[0]。也就是残差 = 预测值 - 观测值
            // 参数维度4，所谓参数的维度是4，就是我们所要拟合的参数C0,C1,C2,C3
            new ceres::AutoDiffCostFunction<CubicCurveResidual, 1, 4>(
                new CubicCurveResidual(x_data[i], y_data[i]));
        /// 添加残差块： cost_function迭代的代价函数
        problem.AddResidualBlock(cost_function, nullptr, params);  // 不使用损失函数
    }

    // ---------- 步骤4：配置求解器选项 ---------- //
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // 使用QR分解
    options.minimizer_progress_to_stdout = true;    // 输出优化过程

    // ---------- 步骤5：运行优化 ---------- //
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ---------- 步骤6：输出结果 ---------- //
    cout << "初始参数: [0, 0, 0, 0]" << endl;
    cout << "优化后参数:\n"
         << "a = " << params[0] << "\n"
         << "b = " << params[1] << "\n"
         << "c = " << params[2] << "\n"
         << "d = " << params[3] << endl;
    cout << "优化报告:\n" << summary.BriefReport() << endl;

    // ---------- 步骤7：可视化结果 ---------- //
    // vector<double> y_fit;
    // for (auto xi : x_data) {
    //     double yi = params[0]*xi*xi*xi + params[1]*xi*xi + params[2]*xi + params[3];
    //     y_fit.push_back(yi);
    // }
    //
    // plt::scatter(x_data, y_data, 10, {{"label", "Noisy Data"}});
    // plt::plot(x_data, y_fit, {{"label", "Ceres Fit"}, {"linewidth", "2"}});
    // plt::xlabel("X");
    // plt::ylabel("Y");
    // plt::title("Cubic Curve Fitting with Ceres");
    // plt::legend();
    // plt::grid(true);
    // plt::show();
}

