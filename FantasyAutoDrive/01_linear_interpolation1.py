


# 线性插值
# 概念：
#         线性插值适用于每对相邻的数据点之间，假设每对点之间的关系是线性的。对于两组数据点集合(x0,y0),(x1,y1),…,(xn,yn)(x0​,y0​),(x1​,y1​),…,(xn​,yn​)，我们可以在每个区间[xi,xi+1][xi​,xi+1​]之间进行线性插值。
#         线性插值是一种最简单的插值方法，适用于两个已知数据点之间的插值。它假设两个数据点之间的函数关系是线性的。对于一维线性插值，如果已知点(x0,y0)(x0​,y0​)和(x1,y1)(x1​,y1​)，则在这两个点之间的某个点xx上的函数值yy可以通过以下公式计算：
#         示例代码（Python）：


def linear_interpolation(xs, ys, x):
    if x < xs[0] or x > xs[-1]:
        raise ValueError(f"x = {x} is outside the interpolation range.")
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

# 示例数据点
xs = [1, 2, 3, 4]
ys = [2, 3, 5, 4]

# 需要插值的点
x = 2.5

# 计算插值
y = linear_interpolation(xs, ys, x)
print(f"The interpolated value at x = {x} is y = {y}")
