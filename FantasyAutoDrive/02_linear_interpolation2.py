def linear_interpolation(x0, y0, x1, y1, x):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

# 示例数据点
x0, y0 = 0.9, 1.9
x1, y1 = 1.1, 3.1

# 需要插值的点
x = 2
x = 3
x = 1

# 计算插值
y = linear_interpolation(x0, y0, x1, y1, x)
print(f"The interpolated value at x = {x} is y = {y}")