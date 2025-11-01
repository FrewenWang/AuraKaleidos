# 代码参考：
# 1.71：图像的低通滤波 (高斯滤波器核)
import cv2
from matplotlib import pyplot as plt

imgFile = "../images/lena_color.tiff"  # 读取文件的路径
img = cv2.imread(imgFile, flags=1)

kSize = (5, 5)
imgGaussBlur1 = cv2.GaussianBlur(img, (5,5), sigmaX=10)
imgGaussBlur2 = cv2.GaussianBlur(img, (11,11), sigmaX=20)

# 计算高斯核
gaussX = cv2.getGaussianKernel(5, 0)
gaussXY = gaussX * gaussX.transpose(1, 0)
print("gaussX:\n", gaussX)
print("gaussXY:\n", gaussXY)

plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132), plt.axis('off'), plt.title("ksize=5, sigma=10")
plt.imshow(cv2.cvtColor(imgGaussBlur1, cv2.COLOR_BGR2RGB))
plt.subplot(133), plt.axis('off'), plt.title("ksize=11, sigma=20")
plt.imshow(cv2.cvtColor(imgGaussBlur2, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()

