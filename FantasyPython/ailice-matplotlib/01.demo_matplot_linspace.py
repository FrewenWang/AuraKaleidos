import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图形和坐标轴
fig, ax = plt.subplots()


# 绘制每个数据点（绘制散点图）
ax.scatter(x, y)

# 绘制折线图（将散点图连成线，绘制成折线图）
ax.plot(x, y)

# 绘制一条竖线，例如在x=5的位置
ax.axvline(x=5, color='r', linestyle='--', lw=2)

# 在竖线上方添加文字
# xy指定文本位置，(5, 1.1)表示文本在x=5, y稍微高于y=sin(5)的位置
# ha='center'表示文本水平居中，va='bottom'表示文本垂直位置在底部，但这里因为文本在上方，所以实际意义不大
# bbox用于添加文本框，facecolor设置填充颜色，alpha设置透明度
ax.text(5, 1.1, 'Vertical Line', ha='center', va='bottom', transform=ax.get_xaxis_transform(),
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# 在折线图上方绘制文本
# 这里选择了一个具体的x和y坐标位置，以及文本内容
# ha='center' 使得文本水平居中，va='bottom' 使得文本垂直对齐方式为底部
# 你可能需要根据你的数据和图表布局来调整这些参数
ax.text(0, 1.2, 'title on left_top', ha='center', va='bottom', fontsize=12, color='red')

# 也可以直接使用坐标轴的x和y坐标范围来定位文本，但这样可能不那么精确
# ax.text(0.5, 1.05, 'Vertical Line', horizontalalignment='center', 
#         verticalalignment='top', transform=ax.transAxes)

# 设置图表标题和轴标签
ax.set_title('Sin Wave with a Vertical Line and Text')
ax.set_xlabel('x')
ax.set_ylabel('sin(x)')

# 显示图表
plt.show()