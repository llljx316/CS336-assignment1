import torch
from torch import Tensor, nn
from einops import rearrange, einsum
from .Linear import Linear

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # self.lq = Linear(d_k, d_k, device)
        # k in {1,2,...,d//2}
        self.sinik_ls = torch.tensor(([[torch.sin(self._calculate_thetaik(i, k)) for k in range(1, self.d_k//2+1)] for i in range(max_seq_len)]), device=device)
        self.cosik_ls = torch.tensor(([[torch.cos(self._calculate_thetaik(i, k)) for k in range(1, self.d_k//2+1)] for i in range(max_seq_len)]), device=device)
        self.register_buffer('sinik', self.sinik_ls, persistent=False)
        self.register_buffer('cosik', self.cosik_ls, persistent=False)#创立缓冲区方便复用
        
        #calculate R
        self.R = torch.stack([self._calculate_ri(i) for i in range(self.max_seq_len)])
        self.register_buffer('Rm', self.R, persistent=False) # [max_seq_len, d_k, d_k]

    def _calculate_thetaik(self, i, k):
        return torch.tensor(i/(self.theta ** ((2*k-2)/self.d_k)), device = self.device)

    def _calculate_ri(self, i):
        ri = torch.zeros(self.d_k, self.d_k, device=self.device)
        for k in range(self.d_k//2):
            ri[k*2:k*2+2, k*2:k*2+2] = self._calculate_rik(i, k)
        return ri

    def _calculate_rik(self, i, k):
        sin_theta_i_k = self.sinik_ls[i][k]
        cos_theta_i_k = self.cosik_ls[i][k]
        return torch.tensor([
            [cos_theta_i_k, -sin_theta_i_k],
            [sin_theta_i_k, cos_theta_i_k],
            ], device=self.device)

    # def reset_parameter(self):
    #     nn.init.trunc_normal_(self.weight)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        #x (..., seq_len, d_k) 
        # token_position (..., seq_len)
        # return a same shape tensor
        # index for correct_R_i
        R_selected = self.R[token_positions]# may have some questions
        # R_selected [..., d_k, d_k]
        return einsum(R_selected, x, '... di dk, ... dk -> ... di') # 注意最终的保留的维度
        

