import torch
from torch import Tensor, nn

def softmax(x: Tensor, i: int):
    # 在维度i进行softmax
    max_value = torch.max(x, dim=i, keepdim=True).values
    new_x=x-max_value
    exp_value = torch.exp(new_x) # trick for stability
    sum_value = torch.sum(exp_value, dim=i, keepdim=True)
    return exp_value/ sum_value

