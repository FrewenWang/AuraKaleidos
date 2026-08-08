import numpy as np
import matplotlib.pyplot as plt




# ====================== 算法实现 ====================== #
def least_squares_fit(x, y):
    """
    最小二乘法拟合三次曲线
    模型：y = a*x³ + b*x² + c*x + d
    """
    # 构建设计矩阵 X: [x³, x², x, 1]
    X = np.column_stack([x ** 3, x ** 2, x, np.ones_like(x)])
    # 最小二乘解：Xθ = y → θ = (X.T X)^-1 X.T y
    theta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return theta  # 返回参数 [a, b, c, d]


def ransac_fit(x, y, max_iters=100, threshold=0.5, min_samples=4):
    """
    RANSAC算法拟合三次曲线
    参数：
        max_iters: 最大迭代次数
        threshold: 内点判定阈值（误差小于此值视为内点）
        min_samples: 每次迭代随机采样的最小点数
    """
    best_inliers = None
    best_theta = None
    best_inlier_count = 0

    for _ in range(max_iters):
        # 1. 随机采样min_samples个点
        sample_indices = np.random.choice(len(x), min_samples, replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]

        # 2. 用采样点拟合模型
        try:
            X_sample = np.column_stack([x_sample ** 3, x_sample ** 2, x_sample, np.ones_like(x_sample)])
            theta = np.linalg.lstsq(X_sample, y_sample, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue  # 忽略奇异矩阵错误

        # 3. 计算所有点的误差
        y_pred = theta[0] * x ** 3 + theta[1] * x ** 2 + theta[2] * x + theta[3]
        errors = np.abs(y_pred - y)
        inliers = errors < threshold  # 布尔掩码，标记内点

        # 4. 更新最佳模型（内点最多）
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_theta = theta
            best_inliers = inliers

    # 5. 用所有内点重新拟合最终模型
    if best_inliers is not None and np.sum(best_inliers) >= min_samples:
        X_inliers = np.column_stack([x[best_inliers] ** 3, x[best_inliers] ** 2,
                                     x[best_inliers], np.ones_like(x[best_inliers])])
        theta = np.linalg.lstsq(X_inliers, y[best_inliers], rcond=None)[0]
        return theta, best_inliers
    else:
        return None, None  # 拟合失败


# ====================== 数据生成 ====================== #
def generate_data(seed=123, outlier_ratio=0.1):
    """生成带噪声和异常值的测试数据"""
    np.random.seed(seed)
    x = np.linspace(-3, 5, 100)
    # 真实模型：y = 0.5x³ - 2x² + x + 3
    y_true = 0.5 * x ** 3 - 2 * x ** 2 + x + 3
    # 添加高斯噪声
    noise = np.random.normal(0, 0.5, size=x.shape)
    # 添加异常值（随机替换部分点为噪声）
    outlier_mask = np.random.rand(len(x)) < outlier_ratio
    y = y_true + noise + outlier_mask * np.random.normal(0, 10, size=x.shape)
    return x, y, y_true


# ====================== 可视化 ====================== #
def plot_results(x, y, y_true, ls_theta, ransac_theta, inliers=None):
    plt.figure(figsize=(10, 6))
    # 原始数据点
    plt.scatter(x, y, c='blue', s=20, label='Data (with outliers)')
    # 标记RANSAC内点
    if inliers is not None:
        plt.scatter(x[inliers], y[inliers], c='green', s=30,
                    label='RANSAC Inliers', edgecolors='k')
    # 真实模型
    plt.plot(x, y_true, 'k--', lw=2, label='True Model')
    # 最小二乘拟合
    if ls_theta is not None:
        y_ls = ls_theta[0] * x ** 3 + ls_theta[1] * x ** 2 + ls_theta[2] * x + ls_theta[3]
        plt.plot(x, y_ls, 'r-', lw=2, label='Least Squares Fit')
    # RANSAC拟合
    if ransac_theta is not None:
        y_ransac = ransac_theta[0] * x ** 3 + ransac_theta[1] * x ** 2 + ransac_theta[2] * x + ransac_theta[3]
        plt.plot(x, y_ransac, 'g-', lw=2, label='RANSAC Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Cubic Curve Fitting Comparison')
    plt.grid(True)
    plt.show()


# ====================== 主程序 ====================== #
if __name__ == "__main__":
    # 生成数据
    x, y, y_true = generate_data(outlier_ratio=0.1)

    # 最小二乘法拟合
    ls_theta = least_squares_fit(x, y)
    print("Least Squares Parameters:")
    print(f"a={ls_theta[0]:.3f}, b={ls_theta[1]:.3f}, c={ls_theta[2]:.3f}, d={ls_theta[3]:.3f}")

    # RANSAC拟合
    ransac_theta, inliers = ransac_fit(x, y, max_iters=200, threshold=1.0)
    if ransac_theta is not None:
        print("\nRANSAC Parameters:")
        print(f"a={ransac_theta[0]:.3f}, b={ransac_theta[1]:.3f}, c={ransac_theta[2]:.3f}, d={ransac_theta[3]:.3f}")
    else:
        print("\nRANSAC failed to fit a model.")

    # 可视化
    plot_results(x, y, y_true, ls_theta, ransac_theta, inliers)