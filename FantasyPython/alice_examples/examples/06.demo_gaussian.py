import numpy as np


def gaussian_kernel(kernel_size, sigma):
    # 生成坐标网络
    center = kernel_size // 2
    x = np.arange(-center, center + 1)
    y = np.arange(-center, center + 1)
    xx, yy = np.meshgrid(x, y)

    # 计算高斯函数值
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / (2 * np.pi * sigma ** 2)  # 归一化系数

    # 确保总和为1
    kernel /= kernel.sum()
    return kernel


def integer_gaussian_kernel(size, sigma):
    kernel = gaussian_kernel(size, sigma)
    kernel *= 256  # 放大系数
    kernel = np.round(kernel).astype(int)
    return kernel


kernel_size = 7
sigma = 1.0
# 示例：生成3x3核，σ=0.8
kernel = gaussian_kernel(kernel_size, sigma)
print("高斯核系数（浮点）:\n", kernel)

# 示例：生成整数化3x3核
int_kernel = integer_gaussian_kernel(kernel_size, sigma)
print("高斯核系数（整数）:\n", int_kernel, ", size:", kernel_size, ", sigma:", sigma)
kernel = gaussian_kernel(kernel_size, sigma)
# 高斯核系数（整数）:
#  [[1 2 1]
#  [2 4 2]
#  [1 2 1]] , size: 3 , sigma: 0.8

# 高斯核系数（整数）:
#  [[1 2 1]
#  [2 3 2]
#  [1 2 1]] , size: 3 , sigma: 1.2

# 高斯核系数（整数）:
#  [[2 2 2]
#  [2 2 2]
#  [2 2 2]] , size: 3 , sigma: 1.9