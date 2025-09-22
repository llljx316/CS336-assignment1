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
        parser.add_argument('--save_dir', type=str, default="./save", help="模型类型")


        vocab_size = 10000
        context_length = 256
        num_layers = 4
        d_model = 512
        d_ff = 1344
        num_heads = 16
        theta = 10000
        lr = 0.01
        weight_decay = 0.01
        batch_size = 32
        # dataset_dir = "../data/TinyStoriesV2-GPT4-train.txt"
        dataset_dir = "./tests/fixtures/tinystories_sample_5M.txt"
        loops = 1000
        # save_dir = "./"


        # device = "mps"
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # device = "cpu"
        # torch.set_default_device(device)
        args = parser.parse_args()
        if args.model == "TransformerLM":
            model = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta).to(device)


        # tokenizer get
        vocab_file_dir = Path(args.save_dir) / "vocab.json"
        merges_file_dir = Path(args.save_dir) / "merges.txt"
        if vocab_file_dir.exists() and merges_file_dir.exists():  
            Tk = tokenizer.from_files(vocab_file_dir, merges_file_dir, special_tokens=["<|endoftext|>"])
        else:
            vocab, merges = train_bpe(dataset_dir, vocab_size, special_tokens=["<|endoftext|>"])
            save_data(args.save_dir, vocab, merges)
            Tk = tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

        f_dataset = open(dataset_dir, 'r')
        tkiter =  Tk.encode_iterable(f_dataset)
        # encode_result = np.array(encode_result)
        # np.save(encode_path, encode_result)
        dataloader = DataLoaderFromIterator(tkiter, batch_size, context_length, device)
        print(next(dataloader))

        # with open(dataset_dir, 'r') as f:
        #     encode_result = []
        #     for _id in Tk.encode_iterable(f):
        #         encode_result.append(_id)

        Tk.encode(' ?!')
        
        # encode_path = Path(args.save_dir)/"save_data.npy"
        # if encode_path.exists():
        #     encode_result = np.load(encode_path)
        # else:
        #     encode_result = np.array(encode_result)
        #     np.save(encode_path, encode_result)