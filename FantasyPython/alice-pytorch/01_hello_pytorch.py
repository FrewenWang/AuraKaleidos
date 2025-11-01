import torch

print(torch.__version__)
print(torch.cuda.is_available())

result = torch.tensor(1) + torch.tensor(2.0)
print(result)

#  测试结果：
# 1.8.0+cu111
# True
# tensor(3.)


# 2.5.1+cu124
# True
# tensor(3.)