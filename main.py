from tests.test_train_bpe import *
import time
import cProfile
import pstats

def test_train_bpe_special_tokens_t():
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes


if __name__=='__main__':
    test_train_bpe()
    #train by tiny story
    # input_path = FIXTURES_PATH / "corpus.en"
    input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    start = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    print(f'{time.time()-start}s.')
    # profiler = cProfile.Profile()
    # profiler.enable()  # 启动性能分析
    # vocab, merges = run_train_bpe(
    #     input_path=input_path,
    #     vocab_size=1000,
    #     special_tokens=["<|endoftext|>"],
    # )
    # profiler.disable()
    # print(f'{time.time()-start}s.')
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')  # 按累计时间排序
    # stats.print_stats(30)  # 打印前10条结果

    # test_train_bpe_speed()
    # test_train_bpe_special_tokens_t()
    # test_train_bpe_special_tokens()