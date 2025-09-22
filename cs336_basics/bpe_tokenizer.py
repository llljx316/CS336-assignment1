from . import PAT
import regex as re
import os
import json
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable
from .pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import torch
# import cProfile
# import re
# cProfile.run('re.compile("foo|bar")')

# def pre_tokenize(text: str, pattern = PAT, special_tokens=None):
    
    
def get_freq(corpus, freq):
    for match in re.finditer(PAT, corpus):
        word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
        freq[word] += 1

def get_freq_without_special_tokens(chunk, freq, special_tokens):
    # 先按 special token 切段，special token 本身作为单一词元 (bytes,) 计数
    if len(special_tokens) == 0:
        get_freq(chunk, freq)
    else:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        special_tokens_re = re.compile("|".join(re.escape(st) for st in special_tokens_sorted)) if special_tokens_sorted else None
        pos = 0
        for sm in special_tokens_re.finditer(chunk):
            if sm.start() > pos:
                get_freq(chunk[pos:sm.start()], freq)
            pos = sm.end()
        if pos < len(chunk):
            get_freq(chunk[pos:], freq)

def process_chunk(chunk, special_tokens):
    freq = defaultdict(int)
    get_freq_without_special_tokens(chunk, freq, special_tokens)
    return freq

def chunk_pre_tokenize(chunk):
    return list(re.findall(PAT, chunk))

def pre_tokenize_without_special_tokens(chunk, special_tokens):
    # 先按 special token 切段，special token 本身作为单一词元 (bytes,) 计数
    if special_tokens is None or len(special_tokens) == 0:
        return chunk_pre_tokenize(chunk)
    else:
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        special_tokens_re = re.compile("|".join(re.escape(st) for st in special_tokens_sorted)) if special_tokens_sorted else None
        res = []
        pos = 0
        for sm in special_tokens_re.finditer(chunk):
            if sm.start() > pos:
                res.extend(chunk_pre_tokenize(chunk[pos:sm.start()]))
            res.append(chunk[sm.start():sm.end()])
            pos = sm.end()
        if pos < len(chunk):
            res.extend(chunk_pre_tokenize(chunk[pos:]))
        return res
    
def pre_tokenize_file(input_path, special_tokens):
    with open(input_path, "rb") as f:
        num_processes = mp.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq = defaultdict(int)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        #special token split
        with mp.Pool(processes=num_processes) as pool:
            # 读取每个数据块并并行处理
            # chunks = [f.read(end - start).decode("utf-8", errors="ignore") for start, end in zip(boundaries[:-1], boundaries[1:])]
            results = pool.starmap(pre_tokenize_without_special_tokens, [(chunk, special_tokens) for chunk in chunks])

        return results

def pre_tokenize_iter(text_iter, special_tokens):
    num_processes = mp.cpu_count()

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    #special token split
    with mp.Pool(processes=num_processes) as pool:
        # 读取每个数据块并并行处理
        # chunks = [f.read(end - start).decode("utf-8", errors="ignore") for start, end in zip(boundaries[:-1], boundaries[1:])]
        results = pool.starmap(pre_tokenize_without_special_tokens, [(chunk, special_tokens) for chunk in text_iter])

    return results

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    
        

    def merge_two_bytes(bl):
        assert len(bl) == 2
        return bl[0]+bl[1]
    


    with open(input_path, "rb") as f:
        num_processes = mp.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq = defaultdict(int)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        #special token split
        with mp.Pool(processes=num_processes) as pool:
            # 读取每个数据块并并行处理
            # chunks = [f.read(end - start).decode("utf-8", errors="ignore") for start, end in zip(boundaries[:-1], boundaries[1:])]
            results = pool.starmap(process_chunk, [(chunk, special_tokens) for chunk in chunks])

        # 合并所有进程的结果
        freq = defaultdict(int)
        for result in results:
            for word, count in result.items():
                freq[word] += count
        freq = dict(freq)

    vocab = {i:bytes([i]) for i in range(256)}
    # add special tokens
    # vocab[len(vocab)] = b"<|endoftext|>"
    for st in special_tokens:
        vocab[len(vocab)]= bytes(st.encode('utf-8'))
    merges = list()

    def change_freq_naive(freq:dict, max_token):
        for word in list(freq.keys()):
            new_word = []
            i = 0
            while i<len(word)-1:
                if (word[i],word[i+1]) == max_token:
                    new_word.append(word[i]+word[i+1])
                    i+=1 
                else:
                    new_word.append(word[i])
                i+=1
            #last one 
            if i<len(word):
                new_word.append(word[i])
                i+=1
            new_word = tuple(new_word)
            if len(new_word) != len(word):
                freq[new_word] = freq.pop(word)

    # 耗时最大
    def change_freq(freq:dict, max_token, pair_storage, pair_to_word):
        # for word, count in list(freq.items()):
        for word in list(pair_to_word[max_token]):
            if word not in freq:
                pair_to_word[max_token].remove(word)#维护 
                continue #不维护可能有浪费，但终究剪纸
            new_word = []
            i = 0
            while i<len(word)-1:
                if (word[i],word[i+1]) == max_token:
                    new_word.append(word[i]+word[i+1])
                    count = freq[word]
                    #update pair_storage
                    if i>0:
                        pair_storage[(word[i-1], word[i]+word[i+1])] += count
                        pair_storage[(word[i-1], word[i])]-=count

                    if i+2<len(word):
                        pair_storage[(word[i]+word[i+1], word[i+2])] += count 
                        pair_storage[(word[i+1], word[i+2])] -= count
                    i+=1 
                else:
                    new_word.append(word[i])
                i+=1
            #last one 
            if i<len(word):
                new_word.append(word[i])
                i+=1
            new_word = tuple(new_word)
            if len(new_word) != len(word):
                freq[new_word] = freq.pop(word)
                #all update
                for i in range(len(new_word)-1):
                    pair_to_word[(new_word[i], new_word[i+1])].add(new_word)
        #update pair storage
        pair_storage.pop((max_token[0], max_token[1]))

    def merge(freq: dict, vocab, merges, vocab_size, special_tokens: set):
        # naive merge
        pair_storage = defaultdict(int)
        pair_to_word = defaultdict(set)
        # pair_index = defaultdict(list)
        #init pair storage and create index
        for word, count in freq.items():
            for i in range(len(word)-1):
                #merge two bytes
                pair_storage[(word[i],word[i+1])]+= count
                pair_to_word[(word[i],word[i+1])].add(word)
                # pair_index[(word[i], word[i+1])].append((word, i)) # use address can do 

        while len(vocab) < vocab_size:
            #find
            # for word, count in freq.items():
            #     for i in range(len(word)-1):
            #         #merge two bytes
            #         #maybe optimized
            #         pair_storage[(word[i],word[i+1])]+= count
            #add merges and vocab 
            #find max value in pair storage
            max_key = max(pair_storage, key=lambda k: (pair_storage[k], k)) # 按照字典序来
            merges.append(max_key)
            vocab[len(vocab)]=merge_two_bytes(max_key)
            #exchange
            change_freq(freq, max_key, pair_storage, pair_to_word)
            

    merge(freq, vocab, merges, vocab_size, special_tokens)



    return vocab, merges

def save_data(save_dir, vocab, merges):
    def bytes_to_unicode_escape(byte_value):
        # return ''.join(f'\\u{byte:04x}' if byte > 127 else chr(byte) for byte in byte_value)
        return ''.join(chr(byte) for byte in byte_value)

    def bytes_to_merges_unicode_escape(byte_value):
        return ''.join(chr(byte) if byte != ord(' ') else 'Ġ' for byte in byte_value)

        # try: 
        #     return byte_value.decode('ascii')
        # except:
        #     ret = ''
        #     for byte_v in byte_value:
        #         ret += 
        #     return f'\\u{byte_value[0]:04x}'
        
    root_dir = Path(save_dir)
    with open(root_dir / "vocab.json", 'w', encoding='utf-8') as f:
        json.dump({str(k): bytes_to_unicode_escape(v) for k, v in vocab.items()}, f)

    with open(root_dir / "merges.txt", 'w', encoding='utf-8') as f:
        # for a, b in merges:
        f.writelines(f"{bytes_to_merges_unicode_escape(a)} {bytes_to_merges_unicode_escape(b)}\n" for a, b in merges)


def merge_op(token, bytes_tokens, p1, p2, index):
    i = 0
    new_token = []
    while i < len(token)-1:
        if token[i]==p1 and token[i+1]==p2:
            new_token.append(p1+p2)
            i+=1
        else:
            new_token.append(token[i])
        i+=1

    if i < len(token):
        new_token.append(token[i])
        i+=1
    if len(token)!=len(new_token):
        bytes_tokens[index] = new_token


class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab_rev = vocab
        self.vocab = {v:k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        self._decode_cache = b''

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            vocab = {int(k): bytes(v, 'utf-8') for k, v in vocab_data.items()}

        def read_merges(file_path):
            merges = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 去掉行首尾的空白字符，并去掉引号
                    parts = line.strip().split(' ')
                    if len(parts) == 2:
                        # 将两个部分转换为元组并添加到 merges 列表中
                        merges.append((parts[0].replace('Ġ', ' ').encode('utf-8'), parts[1].replace('Ġ', ' ').encode('utf-8')))
            return merges

        # with open(merges_filepath, 'r', encoding='utf-8') as f:
            # merges_data = [tuple(line.rstrip().split(" ")) for line in f]
        merges = read_merges(merges_filepath)

        return cls(vocab, merges, special_tokens)
    
    def _encode_str(self, tokens: list[str]):
        #transform token to bytes
        # bytes_tokens = [c.encode() for token in tokens for c in token]
        # bytes_tokens = [bytes([b]) for token in tokens for b in token.encode("utf-8")] 
        bytes_tokens = []
        for token in tokens:
            if self.special_tokens is not None and token in self.special_tokens:
                bytes_tokens.append([token.encode("utf-8")])
            else:
                bytes_tokens.append([bytes([b]) for b in token.encode()])

        #先进行merge

    
        # for p1, p2 in tqdm(self.merges):
        for p1, p2 in self.merges:
            for index, token in enumerate(bytes_tokens):
                merge_op(token, bytes_tokens, p1, p2, index)



        #然后查词
        result = []
        for token in bytes_tokens:
            for bt in token:
                try:
                    result.append(self.vocab[bt])
                except:
                    # try:
                    bt = bt.decode('latin-1').encode('utf-8')
                    result.append(self.vocab[bt])
        return result

    # def _yield_mt(self, token_list):
    #     encode_result = self._encode_str(token_list)
    #     for id in encode_result:
    #         yield id

    def encode(self, text: str) -> list[int]:
        pre_tokenize_result = pre_tokenize_without_special_tokens(text, self.special_tokens)
        return self._encode_str(pre_tokenize_result)

    # def encode(self, )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        if Path('./pre_token_tiny_stories.pth').exists():
            pre_tokenize_result = torch.load('./pre_token_tiny_stories.pth')
        else:
            pre_tokenize_result = pre_tokenize_iter(iterable, self.special_tokens)
            # with open('./pre_token_tiny_stories.npy', "wb") as f:
            torch.save(pre_tokenize_result, './pre_token_tiny_stories.pth')

        # for token_list in tqdm(pre_tokenize_result):
        #     encode_result = self._encode_str(token_list)
            # for id in encode_result:
            #     yield id

        process_num = mp.cpu_count()
        with mp.Pool(process_num) as pool:
            # self._encode_str 是每个工作进程要调用的函数
            # pre_tokenized_chunks 是要分发给各个进程的数据
            # tqdm 用于显示处理进度
            results_iterator = pool.imap_unordered(self._encode_str, tqdm(pre_tokenize_result))
            
            # 3. 从结果迭代器中逐个产出编码后的ID
            for id_list in results_iterator:
                yield from id_list
        # for id in encode_result:
        #     yield id


    def decode(self, ids: list[int]) -> str:
        #查表
        result = self._decode_cache
        self._decode_cache= b''
        for id in ids:
            result += self.vocab_rev[id] 
        try:
            result = result.decode()
        except:
            self._decode_cache = result
        return result

    