import torch
from torch import nn

def save_checkpoint(model, optimizer, iteration, out):
    save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    with open(out, 'wb') as f:
       torch.save(save, f) 

    
def load_checkpoint(src, model, optimizer):
    with open(src, 'rb') as f:
        save = torch.load(f)

    model.load_state_dict(save['model'])
    optimizer.load_state_dict(save['optimizer'])
    return save["iteration"]
