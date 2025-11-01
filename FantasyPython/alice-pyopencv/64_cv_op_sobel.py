# 1.79：图像锐化：Sobel 算子
# img = cv2.imread("../images/Fig0338a.tif", flags=0)  # NASA 月球影像图
import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread("../images/lena_color.tiff", flags=0)

# 使用函数 filter2D 实现 Sobel 算子
kernSobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # SobelX kernel
kernSobelY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # SobelY kernel
imgSobelX = cv2.filter2D(img, -1, kernSobelX, borderType=cv2.BORDER_REFLECT)
imgSobelY = cv2.filter2D(img, -1, kernSobelY, borderType=cv2.BORDER_REFLECT)

# 使用 cv2.Sobel 实现 Sobel 算子
SobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
SobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
absX = cv2.convertScaleAbs(SobelX)  # 转回 uint8
absY = cv2.convertScaleAbs(SobelY)  # 转回 uint8
SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根

plt.figure(figsize=(10, 6))
plt.subplot(141), plt.axis('off'), plt.title("Original")
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(142), plt.axis('off'), plt.title("SobelX")
plt.imshow(SobelX, cmap='gray', vmin=0, vmax=255)
# plt.imshow(imgSobelX, cmap='gray', vmin=0, vmax=255)
plt.subplot(143), plt.axis('off'), plt.title("SobelY")
plt.imshow(SobelY, cmap='gray', vmin=0, vmax=255)
# plt.imshow(imgSobelY, cmap='gray', vmin=0, vmax=255)
plt.subplot(144), plt.axis('off'), plt.title("SobelXY")
plt.imshow(SobelXY, cmap='gray')
plt.tight_layout()
plt.show()
