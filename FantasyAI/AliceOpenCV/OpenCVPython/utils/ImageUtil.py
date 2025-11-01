import numpy as np


class ImageUtil:
    @staticmethod
    def opencv_2value(img):
        img = img.astype(np.float)
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()

        # Grayscale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        # Binarization
        th = 128
        out[out < th] = 0
        out[out >= th] = 255
        return out

    @staticmethod
    def average_pooling(img):
        """
        将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值。
        我们将这种把图片使用均等大小网格分割，并求网格内代表值的操作称为池化（Pooling）。
        池化操作是卷积神经网络（Convolutional Neural Network）中重要的图像处理方式。
        平均池化按照下式定义：
        v = 1/|R| * Sum_{i in R} v_i
        """
        # Average Pooling
        out = img.copy()
        H, W, C = img.shape
        G = 8
        Nh = int(H / G)
        Nw = int(W / G)
        for y in range(Nh):
            for x in range(Nw):
                for c in range(C):
                    out[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.mean(
                        out[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int)
        return out
