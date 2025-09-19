import torch
from torch import Tensor, nn
from .RMSNorm import RMSNorm
from .multihead_self_attention import multihead_self_attention
from .FeedForward import FFN


class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta:float = 10000):
        super().__init__()
        self.RMS1 = RMSNorm(d_model)
        self.RMS2 = RMSNorm(d_model)
        self.causal_mha = multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        self.position_FFN = FFN(d_model, d_ff)



    def forward(self, x):
        token_position = torch.arange(x.shape[-2])#.view(1,-1) # token position 是一个二维的，要对应维度才能够识别seq的位置并且正确的变换
        y1 = x + self.causal_mha(self.RMS1(x), token_position)
        y2 = y1 + self.position_FFN(self.RMS2(y1))
        return y2
        
