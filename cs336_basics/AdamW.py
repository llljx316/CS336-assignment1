import torch
from torch import nn, Tensor
import math
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas = (0.9, 0.999), eps = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        default = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, default)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 1) # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros(p.data.shape))
                v = state.get("v", torch.zeros(p.data.shape))
                grad = p.grad.data # Get the gradient of loss with respect to p.
                m = betas[0] *m + (1-betas[0])*grad
                v = betas[1] *v + (1-betas[1])* (grad**2)
                lrt = lr*math.sqrt(1-(betas[1]**t))/(1-(betas[0]**t))
                p.data -= lrt*m/(torch.sqrt(v)+eps) #update
                p.data -= lr* weight_decay * p.data# weight decay
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v

        return loss