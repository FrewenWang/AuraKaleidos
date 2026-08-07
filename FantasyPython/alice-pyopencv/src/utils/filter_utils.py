import numpy as np


# 在图片上生成椒盐噪声
def add_peppersalt_noise(image, n=10000):
    result = image.copy()
    # 测量图片的长和宽
    w, h = image.shape[:2]
    # 生成n个椒盐噪声
    for i in range(n):
        # 在的宽度和高度的范围内随机生成随机位置
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        # 然后在这个随机位置生成0或者1这两个像素值，然后将这个像素值设置为0或者1
        if np.random.randint(0, 2) == 0:
            result[x, y] = 0
        else:
            result[x, y] = 255
    return result
