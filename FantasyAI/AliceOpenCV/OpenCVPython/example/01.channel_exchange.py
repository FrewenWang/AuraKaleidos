# RGB代表红绿蓝。大多数情况下，RGB颜色存储在结构或无符号整数中，
# 蓝色占据最不重要的“区域”（32位和24位格式的字节），绿色第二少，红色第三少。

# BGR是相同的，除了区域顺序颠倒。红色占据最不重要的区域，绿色占第二位（静止），蓝色占第三位。

# 早期开发者使用BGR作为颜色的空间的原因在于：那个时候的BGR格式在相机制造厂商和软件提供商之间比较受欢迎。
# 例如。在Windows中，当使用 COLORREF 指定颜色值时，使用BGR格式0x00bbggrr。

# 在opencv中，我们来看下如何读取图像，然后将 RGB 通道替换成 BGR 通道这样的Opencv通道交换。

# 下面的代码用于提取图像的红色通道。注意，cv2.imread() 的系数是按 BGR 顺序排列的！其中的变量 red 表示的是仅有原图像红通道的 imori.jpg。

import cv2

img = cv2.imread("../res/image.jpg")
# opencv的图片第一通道是B通道，也就是蓝色分量图像
# opencv的图片第一通道是G通道，也就是绿色分量图像
# opencv的图片第一通道是R通道，也就是红色分量图像
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# RGB > BGR
img[:, :, 0] = r
img[:, :, 1] = g
img[:, :, 2] = b

# Save result
cv2.imwrite("../res/out.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
