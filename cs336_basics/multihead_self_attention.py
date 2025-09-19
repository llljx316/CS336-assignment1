import torch
from torch import Tensor, nn
from einops import rearrange
from .Linear import Linear
from .scaled_dot_product_attention import scaled_dot_product_attention
from .RoPE import RoPE



class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = -1, theta:float = 0.8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model//num_heads
        self.dv = d_model//num_heads
        self.wq = Linear(d_model, num_heads*self.dk)
        self.wk = Linear(d_model, num_heads*self.dk)
        self.wv = Linear(d_model, num_heads*self.dv)
        self.wo = Linear(num_heads*self.dv, d_model)
        if max_seq_len != -1:
            self.rope = RoPE(theta, self.dk, max_seq_len)

    def forward(self, x, token_position = None):
        seqn = x.shape[-2]
        wqx = self.wq(x)# seqn h*dk
        wkx = self.wk(x)# seqn h*dk
        wvx = self.wv(x) # seqn h*dv
        wqx = rearrange(wqx, "... seqn (h dk) -> ... h seqn dk", h=self.num_heads)
        wkx = rearrange(wkx, "... seqn (h dk) -> ... h seqn dk", h=self.num_heads)
        wvx = rearrange(wvx, "... seqn (h dk) -> ... h seqn dk", h=self.num_heads)
        #qk rope
        if token_position is not None:
            wqx = self.rope(wqx, token_position)
            wkx = self.rope(wkx, token_position)
        mask = (1-torch.triu(torch.ones(wqx.shape[:-2]+(seqn, seqn)), diagonal=1)).to(torch.bool)
        r1 = scaled_dot_product_attention(wqx, wkx, wvx, mask)
        r1 = rearrange(r1, "... h seqn dv -> ... seqn (h dv)") # 需要进行head转换
        r2 = self.wo(r1)
        return r2