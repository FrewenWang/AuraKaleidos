

import numpy as np
import cv2

## 读取图片并转换成灰度图
img = cv2.imread("/home/wangzhijiang/02.ProjectSpace/mage20/gaussian_result.jpg",0)

##显示图片
cv2.imshow("img",img)
cv2.waitKey()
