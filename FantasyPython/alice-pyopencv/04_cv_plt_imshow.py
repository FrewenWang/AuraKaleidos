# 1.10 图像显示(plt.imshow)
import cv2
from matplotlib import pyplot as plt
import matplotlib
import platform


imgFile = "../images/lena_color.tiff"  # 读取文件的路径
imgBRG = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)

# 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
imgRGB = cv2.cvtColor(imgBRG, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(imgBRG, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> Gray


# 判断系统平台并设置字体
if platform.system() == 'Darwin':  # macOS
    matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
elif platform.system() == 'Windows':
    matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
else:  # Linux
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK SC'  # 需手动安装

plt.subplot(221), plt.title("1. RGB 格式(mpl)"), plt.axis('off')
plt.imshow(imgRGB)  # matplotlib 显示彩色图像(RGB格式)
plt.subplot(222), plt.title("2. BGR 格式(OpenCV)"), plt.axis('off')
plt.imshow(imgBRG)    # matplotlib 显示彩色图像(BGR格式)
plt.subplot(223), plt.title("3. 设置 Gray 参数"), plt.axis('off')
plt.imshow(imgGray, cmap='gray')  # matplotlib 显示灰度图像，设置 Gray 参数
plt.subplot(224), plt.title("4. 未设置 Gray 参数"), plt.axis('off')
plt.imshow(imgGray)  # matplotlib 显示灰度图像，未设置 Gray 参数
plt.show()
