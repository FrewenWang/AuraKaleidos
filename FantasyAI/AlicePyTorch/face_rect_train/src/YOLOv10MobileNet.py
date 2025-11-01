import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2


class YOLOv10MobileNet(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOv10MobileNet, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.features = mobilenet.features

        # 定义PPYOLO Tiny FPN
        self.fpn = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.yolo_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes * 5, kernel_size=1)  # 4坐标 + 1置信度
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fpn(x)
        x = self.yolo_head(x)
        return x
