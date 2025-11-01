import cv
# 代码参考：https://blog.csdn.net/youcans/article/details/121870848
# 1.70：图像的低通滤波 (盒式滤波器核)
import cv2
from matplotlib import pyplot as plt
import numpy as np
from utils import cv_utils
from utils import filter_utils

# img = cv2.imread("../images/lena_color.tiff", flags=0)  # # flags=0 读取为灰度图像
#
# kSize = (5, 5)
# kernel1 = np.ones(kSize, np.float32) / (kSize[0]*kSize[1])      # 生成归一化盒式核
# imgConv1 = cv2.filter2D(img, -1, kernel1)                       # cv2.filter2D 方法
# imgConv2 = cv2.blur(img, kSize)                                 # cv2.blur 方法
# imgConv3 = cv2.boxFilter(img, -1, kSize)                        # cv2.boxFilter 方法 (默认normalize=True)
#
# # 打印图片素有的像素点
# print("========img============")
# print(img)
# print("========imgConv1============")
# print(imgConv1)
# print("========imgConv2============")
# print(imgConv2)
# print("========imgConv3============")
# print(imgConv3)
#
# print("比较 cv2.filter2D 与 cv2.blur 方法结果相同吗？\t", (imgConv1 == imgConv2).all())
# print("比较 cv2.blur 与 cv2.boxFilter 方法结果相同吗？\t", (imgConv2 == imgConv3).all())
#
# kSize = (11, 11)
# imgConv11 = cv2.blur(img, kSize)  # cv2.blur 方法
#
# plt.figure(figsize=(9, 6))
# plt.subplot(131), plt.axis('off'), plt.title("Original")
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.subplot(132), plt.axis('off'), plt.title("cv2.blur (kSize=[5,5])")
# plt.imshow(imgConv2, cmap='gray', vmin=0, vmax=255)
# plt.subplot(133), plt.axis('off'), plt.title("cv2.blur (kSize=[11,11])")
# plt.imshow(imgConv11, cmap='gray', vmin=0, vmax=255)
# plt.tight_layout()
# plt.show()
img = cv2.imread("../images/lena_color.tiff")
if img is None:
    print('Failed to read the image')

img1 = filter_utils.add_peppersalt_noise(img)
cv2.imshow('img', img1)

# 默认为规定尺寸的1/n的全1矩阵
img2 = cv2.blur(img1, (3, 3))
cv2.imshow('img2', img2)

# boxfilter方框滤波效果和均值滤波效果一致,默认情况下，对方框滤波后的像素值进行归一化
img3 = cv2.boxFilter(img1, -1, (3, 3))
cv2.imshow('img3', img3)

img4 = cv2.boxFilter(img1, -1, (3, 3), normalize=0)
cv2.imshow('img4', img4)

cv2.waitKey(0)
cv2.destroyAllWindows()
