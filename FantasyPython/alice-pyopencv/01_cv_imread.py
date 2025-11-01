import cv2
import numpy as np

# 文章参考：https://blog.csdn.net/justin18chan/article/details/127848515

# 1.1 图像的读取
imgFile = "../images/lena_color.tiff"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
img2 = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像
cv2.imshow("img1", img1) 
cv2.imshow("img2", img2) 


# 1.2 从网络读取图像
# import urllib.request as request
# response = request.urlopen("https://profile.csdnimg.cn/8/E/F/0_youcans")
# imgUrl = cv2.imdecode(np.array(bytearray(response.read()), dtype=np.uint8), -1)
# cv2.imshow("imgUrl", imgUrl)

# 1.3 读取中文路径的图像
imgFile = "../images/lena_color中文名称.tiff"  # 带有中文的文件路径和文件名
# imread() 不支持中文路径和文件名，读取失败，但不会报错!
# img = cv2.imread(imgFile, flags=1)
# 使用 imdecode 可以读取带有中文的文件路径和文件名
img_file = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), -1)
cv2.imshow("imgFile", img_file)


key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭