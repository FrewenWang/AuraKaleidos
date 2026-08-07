import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

"""
视频学习: https://www.bilibili.com/video/BV1hE411t7RN?p=18&vd_source=53cacf0a03cd68cb26255546081fd9a0

# 卷积操作
# https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
# conv = Conv2d(in_channels=3, out_channels=6,
#               kernel_size=3, stride=1, padding=0)
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

out_channels: 输出图像的通道数
kernel_size = (3, 3)
stride = (1, 1)
padding = (0, 0)

"""



dataset = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
