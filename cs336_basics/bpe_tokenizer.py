from . import PAT
import regex as re
import os
import json
from pathlib import Path
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from tqdm import tqdm
import multiprocessing as mp
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
        f.writelines(f'"{bytes_to_unicode_escape(a)}" "{bytes_to_unicode_escape(b)}"\n' for a, b in merges)

class tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

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
                    parts = re.findall(r'"([^"]+)"', line)
                    if len(parts) == 2:
                        # 将两个部分转换为元组并添加到 merges 列表中
                        merges.append((parts[0].strip('"').encode('utf-8'), parts[1].strip('"').encode('utf-8')))
            return merges

        # with open(merges_filepath, 'r', encoding='utf-8') as f:
            # merges_data = [tuple(line.rstrip().split(" ")) for line in f]
        merges = read_merges(merges_filepath)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    