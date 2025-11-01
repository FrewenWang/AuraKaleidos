import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例轨迹点
np.random.seed(0)
num_points = 20
x_orig = np.linspace(0, 10, num_points)
y_orig = np.sin(x_orig) + 0.1 * np.random.randn(num_points)

# 定义优化变量
x = cp.Variable(num_points)
y = cp.Variable(num_points)

# 定义目标函数：最小化曲率变化
# 我们将使用二阶差分（即二阶导数离散近似）来最小化曲率变化
objective = cp.Minimize(
    cp.sum_squares(x[2:] - 2 * x[1:-1] + x[:-2]) +
    cp.sum_squares(y[2:] - 2 * y[1:-1] + y[:-2])
)

# 约束：起点和终点固定（可根据需要调整）
constraints = [x[0] == x_orig[0], y[0] == y_orig[0],
               x[-1] == x_orig[-1], y[-1] == y_orig[-1]]

# 创建问题
problem = cp.Problem(objective, constraints)

# 求解问题
problem.solve()

# 输出结果
print(f"Optimal value: {problem.value}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(x_orig, y_orig, 'o-', label='Original Points')
plt.plot(x.value, y.value, 'ro-', label='Smoothed Path')
plt.xlabel('x')
plt.ylabel('y')
plt.title('QP Smoothed Lane Path')
plt.legend()
plt.grid(True)
plt.show()