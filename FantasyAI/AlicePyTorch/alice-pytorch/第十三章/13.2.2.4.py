import torch
from torch.distributions import Categorical

action_probs = torch.tensor([0.3,0.7])
#根据概率建立一个分布并抽样
dist = Categorical(action_probs)
dist_entropy = dist.entropy()
print("dist_entropy:",dist_entropy)
#自己定义的entropy实现
def entropy(data):
    min_real = torch.min(data)
    logits = torch.clamp(data,min=min_real)
    p_log_p = logits * torch.log(data)
    return -p_log_p.sum(-1)

print("self_entropy:",entropy(action_probs))
