import torch
import numpy.typing as npt

class DataLoader:
    def __init__(self, dataset: npt.NDArray, batch_size, context_length, device='cuda'):
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device 
        self.dataset = dataset

    def __call__(self):
        for data in self.dataset:
            yield torch.tensor(data, device=self.device)
