import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

# 读取目标图像
img = cv2.imread("/image.jpg")
# 显示读取的图像
plt.imshow(img)
pylab.show()
# 定义卷积核，对图像进行边缘检测。定义一个卷积核,
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])
# 使用OpenCV内置的卷积函数
res = cv2.filter2D(img, -1, kernel)
# 显示卷积后的图像
plt.imshow(res)
pylab.show()
