from . import PAT
import regex as re
import os
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries

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

    special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
    special_tokens_re = re.compile("|".join(re.escape(st) for st in special_tokens_sorted)) if special_tokens_sorted else None

    def get_freq(corpus, freq):
        for match in re.finditer(PAT, corpus):
            word = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            freq[word] += 1
    
    def get_freq_without_special_tokens(chunk, freq):
        # 先按 special token 切段，special token 本身作为单一词元 (bytes,) 计数
        if len(special_tokens) == 0:
            get_freq(chunk, freq)
        else:
            pos = 0
            for sm in special_tokens_re.finditer(chunk):
                if sm.start() > pos:
                    get_freq(chunk[pos:sm.start()], freq)
                pos = sm.end()
            if pos < len(chunk):
                get_freq(chunk[pos:], freq)
        

    def merge_two_bytes(bl):
        assert len(bl) == 2
        return bl[0]+bl[1]

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            #special token split
            get_freq_without_special_tokens(chunk, freq)

        freq = dict(freq)

    vocab = {i:bytes([i]) for i in range(256)}
    # add special tokens
    # vocab[len(vocab)] = b"<|endoftext|>"
    for st in special_tokens:
        vocab[len(vocab)]= bytes(st.encode('utf-8'))
    merges = list()

    def change_freq(freq:dict, max_token):
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

    def merge(freq: dict, vocab, merges, vocab_size, special_tokens: set):
        # naive merge
        while len(vocab) < vocab_size:
            pair_storage = defaultdict(int)
            #find
            for word, count in freq.items():
                for i in range(len(word)-1):
                    #merge two bytes
                    #maybe optimized
                    if word[i] in special_tokens:
                        continue
                    if word[i+1] in special_tokens:
                        continue
                    if word[i]+word[i+1] in special_tokens:
                        continue
                    pair_storage[(word[i],word[i+1])]+= count
            #add merges and vocab 
            #find max value in pair storage
            max_key = max(pair_storage, key=lambda k: (pair_storage[k], k)) # 按照字典序来
            merges.append(max_key)
            vocab[len(vocab)]=merge_two_bytes(max_key)
            #exchange
            change_freq(freq, max_key)
            

    merge(freq, vocab, merges, vocab_size, special_tokens)



    return vocab, merges

