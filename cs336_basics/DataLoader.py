import torch
from torch import Tensor
import numpy.typing as npt
import numpy as np
from typing import Iterable, Iterator

class DataLoader:

    i=0

    def __init__(self, dataset: npt.NDArray, batch_size, context_length, device='cuda'):
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device 
        self.dataset = dataset
        

    def __call__(self) -> tuple[Tensor, Tensor] :
        # 需要循环
        # x=torch.tensor(self.dataset[DataLoader.i:DataLoader.i+self.batch_size*self.context_length], device=self.device).view(self.batch_size, self.context_length)
        # y=torch.tensor(self.dataset[DataLoader.i+1:DataLoader.i+1+self.batch_size*self.context_length], device=self.device).view(self.batch_size, self.context_length)
        # print(DataLoader.i+self.batch_size+1+self.context_length)
        def get_ind(i, bias=0):
            ind = np.arange(i, i+self.context_length)%(len(self.dataset) - self.context_length)+bias
            return ind

        x = torch.stack([torch.tensor(self.dataset[get_ind(i, 0)], device = self.device) for i in range(DataLoader.i, DataLoader.i+self.batch_size)])
        y = torch.stack([torch.tensor(self.dataset[get_ind(i, 1)], device=self.device) for i in range(DataLoader.i, DataLoader.i+self.batch_size)])
        DataLoader.i += 1
        return x, y

class DataLoaderFromIterator:
    """
    从一个整数迭代器创建批次。
    这个加载器按顺序从数据集中读取数据，适用于无法将整个数据集加载到内存的情况。
    """
    def __init__(self, dataset: Iterable[int], batch_size: int, context_length: int, device: str = 'cuda'):
        self.dataset_iter: Iterator[int] = iter(dataset)
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        
        # 内部缓冲区，用于存储从迭代器中读取的数据
        self.buffer = []

    def __iter__(self) -> "DataLoaderFromIterator":
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        # 1. 确定形成一个批次需要多少数据
        # 我们需要 batch_size * context_length 个 token 来组成 x
        required_len = self.batch_size * self.context_length
        
        # 2. 如果缓冲区数据不足，则从源迭代器填充
        while len(self.buffer) < required_len + 1:
            try:
                # 从 dataset 迭代器中获取下一个整数并存入缓冲区
                self.buffer.append(next(self.dataset_iter))
            except StopIteration:
                # 如果源迭代器耗尽，且缓冲区中剩余数据不足以构成一个完整批次，
                # 则停止迭代。
                if len(self.buffer) < self.context_length + 1:
                    raise StopIteration
                # 否则，用剩余数据构成最后一个（可能不完整的）批次
                break
        
        # 如果缓冲区数据仍然不足，说明已经没有数据了
        if len(self.buffer) < self.context_length + 1:
            raise StopIteration

        # 3. 从缓冲区中创建批次
        # 我们创建 batch_size 个连续的序列
        x_list = []
        y_list = []
        
        # 计算当前缓冲区能创建多少个序列
        num_sequences = min(self.batch_size, len(self.buffer) // self.context_length)
        
        if num_sequences == 0:
             raise StopIteration

        for i in range(num_sequences):
            start_idx = i * self.context_length
            end_idx = start_idx + self.context_length
            x_list.append(torch.tensor(self.buffer[start_idx:end_idx], dtype=torch.long))
            y_list.append(torch.tensor(self.buffer[start_idx + 1 : end_idx + 1], dtype=torch.long))

        # 4. 从缓冲区中移除已使用的数据
        # 我们保留一些数据以便下一个批次可以与当前批次重叠，但这里为了简单，我们创建不重叠的批次
        used_data_len = num_sequences * self.context_length
        self.buffer = self.buffer[used_data_len:]

        # 5. 堆叠并移动到设备
        x = torch.stack(x_list).to(self.device)
        y = torch.stack(y_list).to(self.device)

        return x, y 