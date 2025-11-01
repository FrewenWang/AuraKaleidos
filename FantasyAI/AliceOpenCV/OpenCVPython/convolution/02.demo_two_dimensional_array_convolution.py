import numpy as np


# 步骤1：先将卷积核进行180°翻转
def array_rotate_180(matrix):
    new_arr = matrix.reshape(matrix.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(matrix.shape)
    return new_arr


# 步骤2：将翻转后的卷积核中心与输入二维矩阵数组第一个元素对齐，
# 并将相乘之后得到的矩阵所有元素进行求和，得到结果矩阵的第一个元素。
# 如果考虑边缘效应，那么卷积核与输入矩阵不重叠的地方也应进行0填充
def my_2d_conv(matrix, kernel):
    # 对矩阵数组进行深复制作为输出矩阵，而输出矩阵将更改其中参与卷积计算的元素
    new_matrix = matrix.copy()
    m, n = new_matrix.shape  # 输入二维矩阵的行、列数
    p, q = kernel.shape  # 卷积核的行、列数
    kernel = array_rotate_180(kernel)  # 对卷积核进行180°翻转
    # 将卷积核与输入二维矩阵进行卷积计算
    for i in range(1, m):
        for j in range(1, n - 1):
            '''
            卷积核与输入矩阵对应的元素相乘，然后通过内置函数sum()对矩阵求和，并将结果保存为输出矩阵对应元素
            '''
        new_matrix[i, j] = (matrix[(i - 1):(i + p - 1), (j - 1):(j + q - 1)] * kernel).sum()
    return new_matrix


if __name__ == '__main__':
    input = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])  # 示例二维矩阵输入
    kernel = np.array([[1, 0, 1], [-1, -1, -1]])  # 示例卷积核
    print(my_2d_conv(input, kernel))
