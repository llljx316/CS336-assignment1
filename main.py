from tests.test_train_bpe import *
from tests.test_tokenizer import *
from cs336_basics.bpe_tokenizer import *
import time
import cProfile
import pstats


if __name__=='__main__':
    test_unicode_string_with_special_tokens_matches_tiktoken()
    test_roundtrip_single_character()
    # test_train_bpe()
    #train by tiny story
    # input_path = FIXTURES_PATH / "corpus.en"
    # input_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    # start = time.time()



    # vocab, merges = run_train_bpe(
    #     input_path=input_path,
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )
    # print(f'{time.time()-start}s.')
    # save_data('./tokenizer_data', vocab, merges)

    # tokenizer.from_files('tokenizer_data/vocab.json','tokenizer_data/merges.txt', special_tokens=[b"<|endoftext|>"])



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