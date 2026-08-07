import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 定义问题的维度
n = 2

# 创建优化变量
x = cp.Variable(n)

# 定义二次目标函数：0.5 * x^T * P * x + q^T * x
P = np.array([[2, 0], [0, 2]])
q = np.array([-2, -5])

# 定义线性约束：Gx <= h
G = np.array([[-1, 0], [0, -1], [-1, -3]])
h = np.array([0, 0, -10])

# 定义目标函数
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

# 定义约束
constraints = [G @ x <= h]

# 创建问题
problem = cp.Problem(objective, constraints)

# 求解问题
problem.solve()

# 输出结果
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x.value}")

# 可视化结果
x_values = np.linspace(-5, 5, 400)
y_values = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x_values, y_values)
Z = 0.5 * (P[0, 0] * X ** 2 + 2 * P[0, 1] * X * Y + P[1, 1] * Y ** 2) + q[0] * X + q[1] * Y

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(x.value[0], x.value[1], 'ro', label='Optimal Point')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Quadratic Programming Solution')
plt.legend()
plt.grid(True)
plt.show()
