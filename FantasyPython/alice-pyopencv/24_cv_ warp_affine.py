# 1.33 仿射变换: 平移、镜像、旋转 (cv2.warpAffine)

import cv2
from matplotlib import pyplot as plt
import numpy as np


imgFile = "../images/lena_color.tiff"  # 读取文件的路径
img = cv2.imread(imgFile)
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # 初始位置
pts2 = np.float32([[50, 100], [200, 50], [100, 250]])  # 终止位置
MA = cv2.getAffineTransform(pts1, pts2)  # 计算 2x3 变换矩阵 MA
dst = cv2.warpAffine(img, MA, (cols, rows))  # 实现仿射变换

plt.figure(figsize=(9, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title("warpAffine")
plt.show()
