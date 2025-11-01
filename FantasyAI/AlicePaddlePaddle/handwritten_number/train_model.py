import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F


# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 定义第一层卷积。 输出的矩阵大小
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        #  定义第一层最大池化层
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        #  定义第二层卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        #  定义第二层最大池化层
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义第三层卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        #  定义两层全连接层
        self.fc1 = Linear(in_features=120, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)

    def forward(self, x):  # [N,1,28,28]
        x = self.conv1(x)  # [N,6,24,24]
        x = F.sigmoid(x)  # [N,6,24,24]
        x = self.max_pool1(x)  # [N,6,12,12]
        x = F.sigmoid(x)  # [N,6,12,12]
        x = self.conv2(x)  # [N,16,8,8]
        x = self.max_pool2(x)  # [N,16,4,4]
        x = self.conv3(x)  # [N,120,1,1]
        x = paddle.reshape(x, [x.shape[0], -1])  # [N,120]
        x = self.fc1(x)  # [N,64]
        x = F.sigmoid(x)  # [N,64]
        x = self.fc2(x)  # [N,10]
        return x
