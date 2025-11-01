import torch
from torch.distributions import Categorical

action_probs = torch.tensor([0.3,0.7])
#输出不同分布的log值
print(torch.log(action_probs))
#根据概率建立一个分布并抽样
dist = Categorical(action_probs)
action = dist.sample()
#获取抽样结果对应的分布log值
action_logprobs = dist.log_prob(action)
print(action_logprobs)
