import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    """
    我们重点看一下牛顿差值进行差商的计算
    :param x:
    :param y:
    :return:
    """
    #
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef[0, :]

def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

# 示例数据点
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1, 4, 9, 16, 25])

# 计算差商。计算这两数据集合的差商。
coefficients = divided_diff(x_data, y_data)

# 在一组点上评估多项式
x = np.linspace(1, 5, 100)
y = newton_poly(coefficients, x_data, x)


# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'bo', label='Data Points')
plt.plot(x, y, 'r', label='Newton Interpolating Polynomial')
plt.legend()
plt.title('Newton Interpolation')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.grid(True)
plt.show()