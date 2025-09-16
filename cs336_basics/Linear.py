import torch
from torch import Tensor, nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype), requires_grad=True)
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.trunc_normal_(self.weight)

    def forward(self, x:Tensor) -> Tensor:
        return x @ self.weight.T
