import torch

print(torch.__version__)
print(torch.cuda.is_available())

print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
print("GPU名称：", torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称

# 测试结果：
# 1.8.0+cu111
# True
# 是否可用： True
# GPU数量： 1
# torch方法查看CUDA版本： 11.1
# GPU索引号： 0
# GPU名称： NVIDIA RTX A4000
