import torch
import argparse
import numpy as np
from pathlib import Path
from cs336_basics import *


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='training script for TransformerLM')

        parser.add_argument('--model', type=str, default="TransformerLM", help="模型类型")
        parser.add_argument('--vocab_size', type=str, default="TransformerLM", help="模型类型")
        parser.add_argument('--context_length', type=str, default="TransformerLM", help="模型类型")
        parser.add_argument('--num_layers', type=str, default="TransformerLM", help="模型类型")
        parser.add_argument('--checkpoint', type=str, default=None, help="模型类型")
        parser.add_argument('--save_dir', type=str, default="./save/checkpoints/", help="模型类型")


        vocab_size = 100
        context_length = 8
        num_layers = 2
        d_model = 64
        d_ff = 128
        num_heads = 3
        theta = 10000
        lr = 0.01
        weight_decay = 0.01
        batch_size = 32
        dataset_dir = "./"
        loops = 1000
        # save_dir = "./"

        # device = "mps"
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # device = "cpu"
        # torch.set_default_device(device)
        args = parser.parse_args()
        if args.model == "TransformerLM":
            model = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta).to(device)


        optimizer = AdamW(model.parameters(),lr, weight_decay)
        dataset = np.arange(0, 100)
        #load dataset
        # np.memmap()
        dataloader = DataLoader(dataset, batch_size, context_length, device=device)
        start_t = 0 if args.checkpoint is None else load_checkpoint(args.checkpoint, model, optimizer)

        for t in range(loops):
            x, y = dataloader()
            y_t = model(x)
            loss = cross_entropy(y_t, y)
            # if torch.isnan(loss):
            #      cross_entropy(y_t, y)
            loss.backward()
            print(f"loss: {loss}")
            optimizer.step()
            save_checkpoint(model, optimizer, t, Path(args.save_dir) / 'model.pth')

        # return model



            

