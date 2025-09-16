import torch
from torch import Tensor, nn
from .Linear import Linear

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.d_ff = d_ff # better be 3/8*d_model
        self.l1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.l2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.l3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype) 
    
    def _silu(self, x):
        return x*torch.sigmoid(x)
    
    def forward(self, x: Tensor):
        # w1x = self.l1(x)
        # siluw1x = self._silu(w1x)
        # w3x = self.l3(x)
        return self.l2(self._silu(self.l1(x))* self.l3(x))

