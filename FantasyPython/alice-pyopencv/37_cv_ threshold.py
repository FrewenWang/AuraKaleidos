

# 1.47 固定阈值二值变换
import cv2
from matplotlib import pyplot as plt

imgFile = "../images/lena_color.tiff"  # 读取文件的路径
img = cv2.imread(imgFile)  # 读取彩色图像(BGR)
imgGray = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 颜色转换：BGR(OpenCV) -> Gray

# cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
ret1, img1 = cv2.threshold(imgGray, 63, 255, cv2.THRESH_BINARY)         # 转换为二值图像, thresh=63
ret2, img2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)        # 转换为二值图像, thresh=127
ret3, img3 = cv2.threshold(imgGray, 191, 255, cv2.THRESH_BINARY)        # 转换为二值图像, thresh=191
ret4, img4 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)    # 逆二值图像，BINARY_INV
ret5, img5 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TRUNC)         # TRUNC 阈值处理，THRESH_TRUNC
ret6, img6 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO)        # TOZERO 阈值处理，THRESH_TOZERO

plt.figure(figsize=(9, 6))
titleList = ["1. BINARY(thresh=63)", "2. BINARY(thresh=127)", "3. BINARY(thresh=191)", "4. THRESH_BINARY_INV", "5. THRESH_TRUNC", "6. THRESH_TOZERO"]
imageList = [img1, img2, img3, img4, img5, img6]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.title(titleList[i]), plt.axis('off')
    plt.imshow(imageList[i], 'gray')  # 灰度图像 ndim=2
plt.show()
