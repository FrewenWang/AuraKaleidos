import torch

image = torch.randn(size=(5,3,128,128))
# image = torch.nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)(image)
# print(image.shape)
#
#
# image = torch.nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,groups=3)(image)
# print(image.shape)

depth_conv = torch.nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, groups=3)
point_conv = torch.nn.Conv2d(in_channels=3, out_channels=7, kernel_size=1)
depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)


