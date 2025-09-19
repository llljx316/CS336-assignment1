import torch
from torch import Tensor
from .softmax import softmax


def cross_entropy(inputs:Tensor, targets: Tensor):
    # max = inputs.max(dim=-1, keepdim=True)

    # p_res = softmax(inputs, -1)[targets.view(1,-1)] # i+1
    # p_res = softmax(inputs, -1).gather(dim=-1, index=targets.unsqueeze(-1))  # targets 需要扩展维度
    # p_res = torch.clamp(p_res, min=1e-10)
    # loss = -torch.log(p_res).mean()
    # target_logits = inputs[targets.view]
    target_l = inputs.gather(dim=-1, index = targets.unsqueeze(-1))
    sum_exp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    loss = (-target_l + sum_exp).mean()
    return loss

    # # Extract logits corresponding to the target class
    # target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    
    # # log-sum-exp trick for numerical stability by subtracting the largest element
    # logsumexp = torch.logsumexp(inputs, -1, keepdim=True)

    # # Cancel out log and exp after softmax when calculating loss
    # loss_matrix = -target_logits + logsumexp
    
    # # Average loss
    # loss = torch.mean(loss_matrix)
    # return loss
