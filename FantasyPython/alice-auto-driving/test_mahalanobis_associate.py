import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import make_spd_matrix


# plt.rc("font", family='Microsoft YaHei')
# plt.rc("font", family='SimHei')


plt.rc("font", family='Arial Unicode MS')  # MacOS

# ======================
# 生成模拟数据（真实位置 vs 带噪声观测）
# ======================
np.random.seed(42)

# 真实障碍物位置（上一帧）
true_positions = np.array([[1, 2], [4, 5], [7, 8]])

# 观测噪声协方差矩阵（横向噪声大，纵向噪声小）
cov = np.array([[2.0, 0.0], [0.0, 0.2]])  # 马氏距离将自动修正这种差异
# cov = make_spd_matrix(2)  # 随机生成一个协方差矩阵（取消注释尝试不同情况）

# 生成带噪声的观测位置（当前帧）
noise = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=3)
observed_positions = true_positions + noise


# ======================
# 定义距离计算函数
# ======================
def euclidean_distance(x, y):
    """欧式距离：直接计算两点间直线距离"""
    return cdist(x, y, metric='euclidean')


def mahalanobis_distance(x, y, cov_inv):
    """
    传入噪声协方差矩阵的逆
    """
    """马氏距离：修正协方差影响后的距离"""
    delta = x[:, np.newaxis, :] - y[np.newaxis, :, :]  # 计算差值矩阵
    left = np.einsum('...i,...ij->...j', delta, cov_inv)  # 矩阵乘法 (delta * cov^{-1})
    return np.sqrt(np.einsum('...i,...i->...', left, delta))  # 开根号


# ======================
# 执行匹配算法
# ======================
# 计算协方差矩阵的逆（马氏距离所需） 计算马氏距离匹配的协方差矩阵的逆
cov_inv = np.linalg.inv(cov)

# 计算距离矩阵
# 计算欧式距离
dist_euclid = euclidean_distance(observed_positions, true_positions)
dist_mahalanobis = mahalanobis_distance(observed_positions, true_positions, cov_inv)

# 找到每个观测点的最近匹配（欧式 vs 马氏）
matches_euclid = np.argmin(dist_euclid, axis=1)
matches_mahalanobis = np.argmin(dist_mahalanobis, axis=1)

# ======================
# 可视化结果
# ======================
plt.figure(figsize=(12, 6))

# 绘制真实位置和观测位置
plt.scatter(true_positions[:, 0], true_positions[:, 1],
            c='green', s=100, marker='o', label='上一帧真实位置')
plt.scatter(observed_positions[:, 0], observed_positions[:, 1],
            c='red', s=100, marker='x', label='当前帧观测位置')

# 绘制欧式距离匹配连线
for i, j in enumerate(matches_euclid):
    plt.plot([observed_positions[i, 0], true_positions[j, 0]],
             [observed_positions[i, 1], true_positions[j, 1]],
             'b--', lw=1, alpha=0.7, label='欧式距离' if i == 0 else "")

# 绘制马氏距离匹配连线
for i, j in enumerate(matches_mahalanobis):
    plt.plot([observed_positions[i, 0], true_positions[j, 0]],
             [observed_positions[i, 1], true_positions[j, 1]],
             'm-.', lw=1, alpha=0.7, label='马氏距离' if i == 0 else "")

plt.title('障碍物匹配结果对比（欧式 vs 马氏距离）')
plt.xlabel('横向位置 (X)')
plt.ylabel('纵向位置 (Y)')
plt.legend()
plt.grid(True)
plt.show()
