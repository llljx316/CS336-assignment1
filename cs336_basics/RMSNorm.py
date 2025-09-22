import torch
from torch import Tensor, nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.g)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rmsa = torch.sqrt((x ** 2).sum(dim=-1, keepdim=True)/self.d_model + self.eps)
        result = x*self.g/rmsa
        # Return the result in the original dtype
        return result.to(in_dtype)                        