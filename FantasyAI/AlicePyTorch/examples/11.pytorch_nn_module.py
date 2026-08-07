import torch
from torch import nn

# 查看官方文档：https://docs.pytorch.org/docs/stable/nn.html#module-torch.nn

#  



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # 这个前行推理函数的逻辑非常简单就是数据加1
        output = input + 1
        return output


MyModel = MyModel()
x = torch.tensor(1.0)
output = MyModel(x)
print(output)
# 输出结果为：tensor(2.)
# 创建一个nn.Module类，继承nn.Module类，并实现forward()方法，这个方法就是神经网络的前向传播逻辑。
