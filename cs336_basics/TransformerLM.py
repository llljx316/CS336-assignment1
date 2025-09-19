import torch
from torch import Tensor, nn
# from .bpe_tokenizer import tokenizer
from .Embedding import Embedding
from .transformer_block import transformer_block
from .RMSNorm import RMSNorm
from .softmax import softmax
from .Linear import Linear

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, d_ff: int, num_heads:int, theta:float):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blks = nn.Sequential(
            *[transformer_block(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)] ,
        ) 
        self.post_norm = RMSNorm(d_model)
        self.post_linear = Linear(d_model, vocab_size) # transformer back to next distribution
        
    def forward(self, x: Tensor):
        y1 = self.embedding(x)
        y2 = self.transformer_blks(y1)
        y3 = self.post_norm(y2)
        y4 = self.post_linear(y3) # 没有softmax
        return y4
