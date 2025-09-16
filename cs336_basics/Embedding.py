import torch
from torch import nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embedding = num_embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
        # one_hot=torch.nn.functional.one_hot(token_ids, self.num_embedding)
        # return einsum(one_hot.float(), self.weight, "... vocab , vocab embdim -> ... embdim")

    