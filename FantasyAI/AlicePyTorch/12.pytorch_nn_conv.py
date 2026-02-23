import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 卷积核。也就是模型的权重参数
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 改变input和kernel的形状，将输入的图片变成一个batch，一个通道，一个5*5的图片
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)

# 5*5的junk，3*3的kernel，stride=1。卷积之后的结果是3*3(5-2)
output = F.conv2d(input, kernel, stride=1)
print(output)

# 5*5的junk，3*3的kernel，stride=2。卷积之后的结果是2*2
output2 = F.conv2d(input, kernel, stride=2)
print(output2)

# 讲解padding: 
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)

