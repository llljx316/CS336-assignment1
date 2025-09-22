import torch
from torch import Tensor, nn
import math
from .softmax import softmax
from einops import einsum, rearrange

__all__= ['scaled_dot_product_attention']
# Q: Float[Tensor, " ... queries d_k"],
# K: Float[Tensor, " ... keys d_k"],
# V: Float[Tensor, " ... values d_v"],
# mask: Bool[Tensor, " ... queries keys"] | None = None,
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask:Tensor):
    d_k = q.shape[-1]
    new_mask =torch.where(mask, 0, float('-inf')).to(q.device) 
    r0 = einsum(q, k, "... q d_k, ... k d_k -> ... q k")/math.sqrt(d_k) + new_mask
    r1 = softmax(r0, -1)
    r = einsum(r1, v, "... q m, ... m d_v -> ... q d_v")

    return r
