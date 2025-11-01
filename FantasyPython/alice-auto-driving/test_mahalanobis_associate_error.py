import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, mahalanobis

# 设置随机种子确保可重复性
np.random.seed(42)

# 假设障碍物状态为 [x, y, vx, vy]，且位置和速度存在相关性
num_obstacles = 5
true_cov = np.array([
    [1.0, 0.8, 0.5, 0.3],  # x与其他维度的协方差
    [0.8, 1.2, 0.6, 0.4],  # y
    [0.5, 0.6, 0.9, 0.7],  # vx
    [0.3, 0.4, 0.7, 1.0]   # vy
])

# 生成上一帧和当前帧的障碍物状态（带噪声）
prev_frame = np.random.multivariate_normal(
    mean=[0, 0, 1, 1],  # 初始均值 [x, y, vx, vy]
    cov=true_cov,
    size=num_obstacles
)

current_frame = prev_frame + np.random.normal(0, 0.1, size=prev_frame.shape)  # 添加运动噪声



def match_obstacles_euclidean(prev, current):
    """欧式距离匹配"""
    matches = []
    for i, curr_obs in enumerate(current):
        distances = [euclidean(curr_obs[:2], prev_obs[:2]) for prev_obs in prev]  # 仅用位置(x,y)
        best_match = np.argmin(distances)
        matches.append((i, best_match))
    return matches

def match_obstacles_mahalanobis(prev, current, cov_inv):
    """马氏距离匹配（考虑协方差）"""
    matches = []
    for i, curr_obs in enumerate(current):
        # 计算所有历史障碍物的马氏距离（使用完整状态）
        distances = [
            mahalanobis(curr_obs, prev_obs, cov_inv)
            for prev_obs in prev
        ]
        best_match = np.argmin(distances)
        matches.append((i, best_match))
    return matches

# 计算协方差矩阵的逆（实际中应使用估计的协方差）
cov_inv = np.linalg.inv(true_cov)

# 执行匹配
euclidean_matches = match_obstacles_euclidean(prev_frame, current_frame)
mahalanobis_matches = match_obstacles_mahalanobis(prev_frame, current_frame, cov_inv)


plt.figure(figsize=(14, 6))

# 绘制欧式距离匹配结果
plt.subplot(1, 2, 1)
plt.scatter(prev_frame[:, 0], prev_frame[:, 1], c='blue', label='Previous Frame')
plt.scatter(current_frame[:, 0], current_frame[:, 1], c='red', marker='x', label='Current Frame')
for i, j in euclidean_matches:
    plt.plot([prev_frame[j, 0], current_frame[i, 0]],
             [prev_frame[j, 1], current_frame[i, 1]], 'k--', alpha=0.5)
plt.title("Euclidean Distance Matching (Position Only)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()

# 绘制马氏距离匹配结果
plt.subplot(1, 2, 2)
plt.scatter(prev_frame[:, 0], prev_frame[:, 1], c='blue', label='Previous Frame')
plt.scatter(current_frame[:, 0], current_frame[:, 1], c='red', marker='x', label='Current Frame')
for i, j in mahalanobis_matches:
    plt.plot([prev_frame[j, 0], current_frame[i, 0]],
             [prev_frame[j, 1], current_frame[i, 1]], 'g--', alpha=0.8)
plt.title("Mahalanobis Distance Matching (Full State)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()

plt.tight_layout()
plt.show()