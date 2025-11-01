import torch
from torch.distributions import Categorical

action_probs = torch.tensor([0.3,0.7])
#自己定义的entropy实现
def entropy(data):
    min_real = torch.min(data)
    logits = torch.clamp(data,min=min_real)
    p_log_p = logits * torch.log(data)
    return -p_log_p.sum(-1)

print(entropy(action_probs))
