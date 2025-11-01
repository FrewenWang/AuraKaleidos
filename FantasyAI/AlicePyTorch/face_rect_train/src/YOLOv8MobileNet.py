import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2


class YOLOv8MobileNet(nn.Module):
    def __init__(self, num_classes=1):
        super(YOLOv8MobileNet, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        self.features = mobilenet.features

        self.yolo_head = nn.Sequential(
            nn.Conv2d(1280, 1024, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes * 5, kernel_size=1, stride=1)  # 4坐标 + 1置信度
        )

    def forward(self, x):
        x = self.features(x)
        x = self.yolo_head(x)
        return x
