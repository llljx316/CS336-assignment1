import torch
import argparse
import numpy as np
from pathlib import Path
from cs336_basics import *


class eval:
    def __init__(self, Tk: tokenizer, model, dataset_path, batch_size, context_length, device):
        result = []
        with open(dataset_path, "r") as f:
            data_iter = Tk.encode_iterable(f, './pre_token_tiny_stories_valid.pth')
            for id in data_iter:
                result.append(id)
        dataset = np.array(result) 

        self.dataloader = DataLoaderEval(dataset, batch_size, context_length, device)
        self.model = model

    def eval_model(self):
        results = np.array()
        for x, y in self.dataloader:
            y_eval = self.model(x)
            loss = cross_entropy(y_eval, y)
            results.append(loss)

        return results.mean()




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
    lr = 1e-2
    weight_decay = 0.01
    batch_size = 128
    dataset_dir = "../data/TinyStoriesV2-GPT4-train.txt"
    eval_dataset_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # dataset_dir = "./tests/fixtures/tinystories_sample_5M.txt"
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

    
    
    encode_path = Path(args.save_dir)/"save_data.npy"
    if encode_path.exists():
        dataset = np.load(encode_path)
        dataloader = DataLoader(dataset, batch_size, context_length, device=device)
    else:
        f_dataset = open(dataset_dir, 'r')
        tkiter =  Tk.encode_iterable(f_dataset, './pre_token_tiny_stories.pth')
        # encode_result = np.array(encode_result)
        # np.save(encode_path, encode_result)
        dataloader = DataLoaderFromIterator(tkiter, batch_size, context_length, device)

    print("finish saving data")


    optimizer = AdamW(model.parameters(),lr, weight_decay)
    #load dataset
    # np.memmap()
    start_t = 0 if args.checkpoint is None else load_checkpoint(args.checkpoint, model, optimizer)
    eval_loss = float('inf')
    evaluator = eval(Tk, model, eval_dataset_path, batch_size, context_length, device)
    with tqdm(range(start_t, loops), desc="Training") as pbar:
        for t in pbar:
            optimizer.zero_grad()
            x, y = dataloader() if type(dataloader) is DataLoader else dataloader.__next__()
            y_t = model(x)
            loss = cross_entropy(y_t, y)
            # if torch.isnan(loss):
            #      cross_entropy(y_t, y)
            loss.backward()
            optimizer.step()
            # if t%100==0:
            eloss = evaluator.eval_model()
            if eloss < eval_loss:
                save_checkpoint(model, optimizer, t, Path(args.save_dir) / 'model.pth')
                eval_loss = eloss
            pbar.set_postfix(loss=loss.item(), eval_loss=eloss.item())


    f_dataset.close()
    # return model



        