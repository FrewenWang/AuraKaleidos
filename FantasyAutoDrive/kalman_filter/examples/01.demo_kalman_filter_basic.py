import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ========== 1. 参数设置 ==========
dt = 0.1  # 时间步长（秒）
total_time = 10  # 总时长（秒）
num_steps = int(total_time / dt)  # 总步数

# F：状态转移矩阵 F（假设匀速运动模型）
# 车辆的状态向量为 [位置, 速度]
# 位置_new = 位置_old + 速度_old * dt
# 速度_new = 速度_old
F = np.array([[1, dt],
              [0, 1]])
print("================状态转移矩阵 F==============================\n", F)

# 观测矩阵 H（只能观测到位置）
H = np.array([[1, 0]])
print("================状态观测矩阵 H==============================\n", H)

# 过程噪声协方差 Q（模型误差）
Q = np.diag([0.1, 0.1])  # 位置和速度的模型噪声
# TODO 也就是说过程转换的时候，存在10%的误差？？
print("================状态观测矩阵 H==============================\n", Q)

# 观测噪声协方差 R（传感器误差）
R = np.array([[0.5]])  # 假设位置观测噪声方差为0.5

print("================观测噪声协方差 R==============================\n", R)

# 初始状态 [位置, 速度]
true_state = np.array([0.0, 1.0])  # 真实初始状态：位置0，速度1.0 m/s
est_state = np.array([0.0, 0.5])  # 初始估计状态：故意设置不准确（测试收敛）
est_cov = np.diag([10.0, 10.0])  # 初始协方差矩阵（高不确定性）

# 存储数据用于绘图
time = np.arange(0, total_time, dt)
true_positions = []
true_velocities = []
measured_positions = []
estimated_positions = []
estimated_uncertainty = []

# ========== 2. 模拟数据生成 + 卡尔曼滤波处理 ==========
for i in range(num_steps):
    # ----------------------------
    # (1) 生成真实状态和带噪声的观测值
    # ----------------------------
    # 更新真实状态（匀速运动）
    true_state = F @ true_state  # 真实状态 = F * 上一状态   @在python中是矩阵乘法的运算符。
    true_positions.append(true_state[0])
    true_velocities.append(true_state[1])

    # 生成带噪声的观测值（位置）
    measurement = true_state[0] + np.random.randn() * np.sqrt(R[0, 0])
    measured_positions.append(measurement)

    # ----------------------------
    # (2) 卡尔曼滤波五步流程
    # ----------------------------
    # --- 步骤1：预测车辆的下一个状态向量（方程1）---
    # x_k^- = F * x_{k-1}
    # 如果是第一次遍历这个for循环的，est_state就是刚开始的初始化估计状态
    # 后面再次进行计算的时候：est_state上一次卡尔曼滤波执行的之后的估计状态，承伤一个状态
    predicted_state = F @ est_state

    # --- 步骤2：预测协方差矩阵（方程2）---
    # P_k^- = F * P_{k-1} * F^T + Q
    # 状态转移矩阵（F）： 状态转移矩阵。描述系统状态如何随时间演变。例如，若状态包含位置和速度，F 会将当前状态映射到下一时刻的状态（如位置更新为位置+速度×时间）。
    # 当前协方差矩阵（est_cov）： 表示当前状态估计的不确定性。协方差矩阵的对角元素是各状态变量的方差，非对角元素是变量间的协方差。
    # 过程噪声协方差矩阵（Q）： 反映系统模型的不确定性（如未建模的动态特性或外部扰动）。Q通常假设为白噪声，与状态无关且均值为零。
    # 预测协方差矩阵（predicted_cov）： 输出结果，表示预测状态的不确定性。
    predicted_cov = F @ est_cov @ F.T + Q

    # --- 步骤3：计算卡尔曼增益方程（方程3）---
    # K = P_k^- * H^T / (H * P_k^- * H^T + R)
    K_numerator = predicted_cov @ H.T  # 分子：P_k^- * H^T
    K_denominator = H @ predicted_cov @ H.T + R  # 分母：H * P_k^- * H^T + R
    K = K_numerator @ np.linalg.inv(K_denominator)  # 矩阵除法

    # --- 步骤4：更新障碍物的的状态向量（方程4）---
    # x_k = x_k^- + K * (z_k - H * x_k^-)
    residual = measurement - H @ predicted_state  # 观测残差：z - H * x^-
    updated_state = predicted_state + K @ residual

    # --- 步骤5：更新协方差矩阵（方程5）---
    # P_k = (I - K * H) * P_k^-
    I = np.eye(2)  # 单位矩阵
    updated_cov = (I - K @ H) @ predicted_cov

    # 保存结果
    estimated_positions.append(updated_state[0])    # 更新之后的状态向量
    estimated_uncertainty.append(np.sqrt(updated_cov[0, 0]))  # 位置标准差

    # 更新下一轮的估计值和协方差
    est_state = updated_state   # 更新的状态向量
    est_cov = updated_cov       # 更新的协方差矩阵

# ========== 3. 可视化结果 ==========
plt.figure(figsize=(12, 8))

# (a) 绘制位置随时间变化
plt.subplot(2, 1, 1)
plt.plot(time, true_positions, 'b-', label='real_position', linewidth=2)
plt.plot(time, measured_positions, 'rx', label='measured_position', markersize=4, alpha=0.6)
plt.plot(time, estimated_positions, 'g--', label='kalman_filter', linewidth=2)
plt.fill_between(
    time,
    np.array(estimated_positions) - 3 * np.array(estimated_uncertainty),
    np.array(estimated_positions) + 3 * np.array(estimated_uncertainty),
    color='g', alpha=0.1, label='3σ_estimated_position'
)
plt.xlabel('time (seconds)')  # 时间，单位是秒
plt.ylabel('position (miles)')  # 位置 (米)
plt.title('kalman_filter_tracker')  # 卡尔曼滤波位置追踪
plt.legend()
plt.grid(True)

# (b) 绘制速度随时间变化
plt.subplot(2, 1, 2)
plt.plot(time, true_velocities, 'b-', label='real_speed', linewidth=2)  # 真实速度
plt.plot(time, [est_state[1]] * num_steps, 'g--', label='predict_speed', linewidth=2)  # 预测速度
plt.xlabel('time (seconds)')  # 时间，单位是秒
plt.ylabel('speed (m/s)')
plt.title('speed_estimate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
