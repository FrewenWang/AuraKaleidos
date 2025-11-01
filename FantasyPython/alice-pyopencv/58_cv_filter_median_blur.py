# 1.73：图像的非线性滤波 (中值滤波器)
import cv2
from matplotlib import pyplot as plt

imgFile = "../images/barbara_gray.bmp"  # 读取文件的路径
img = cv2.imread(imgFile, flags=1)

imgMedianBlur1 = cv2.medianBlur(img, 3)
imgMedianBlur2 = cv2.medianBlur(img, 7)

plt.figure(figsize=(9, 6))
plt.subplot(131), plt.axis('off'), plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(132), plt.axis('off'), plt.title("cv2.medianBlur(size=3)")
plt.imshow(cv2.cvtColor(imgMedianBlur1, cv2.COLOR_BGR2RGB))
plt.subplot(133), plt.axis('off'), plt.title("cv2.medianBlur(size=7)")
plt.imshow(cv2.cvtColor(imgMedianBlur2, cv2.COLOR_BGR2RGB))
plt.tight_layout()
plt.show()





