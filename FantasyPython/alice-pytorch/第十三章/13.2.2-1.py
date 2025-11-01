import torch
from torch.distributions import Categorical

action_probs = torch.tensor([0.3,0.7])		#人工建立一个概率值
dist = Categorical(action_probs)			#根据概率建立分布
c0 = 0
c1 = 1
for _ in range(10240):
    action = dist.sample()					#根据概率分布进行抽样
    if action == 0:							#对抽样结果进行存储
        c0 += 1
    else:
        c1 += 1
print("c0的概率为：",c0/(c0 + c1))            #打印输出的结果
print("c1的概率为：",c1/(c0 + c1))
