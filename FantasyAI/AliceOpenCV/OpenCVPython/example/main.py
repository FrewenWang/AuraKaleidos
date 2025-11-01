import cv2
import numpy as np
from utils.ImageUtil import ImageUtil

# Read image
img = cv2.imread("../res/image.jpg")

# 使用OpenCV对图像进行二值化处理
# out = ImageUtil.opencv_2value(img)

out = ImageUtil.average_pooling(img)

# Save result
cv2.imwrite("../res/out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
