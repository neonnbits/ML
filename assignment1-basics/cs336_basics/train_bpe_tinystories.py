import time
import pickle
from cs336_basics.bpe import train

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]

if __name__ == "__main__":
    start = time.perf_counter()

    vocab, merges = train("data/TinyStoriesV2-GPT4-train.txt", VOCAB_SIZE, SPECIAL_TOKENS)
    
    end = time.perf_counter()

    with open("data/tinystories_tokenizer.pkl", "wb") as f:
        pickle.dump((vocab, merges), f)

    longest = max(vocab.values(), key=len)
    print(f"token:{longest}, length:{len(longest)}")
    print(f"tokens as string:{longest.decode("utf-8", errors="replace")}")
    print(f"Time taken: {end - start:.6f} seconds")
    print(f"Vocab length: {len(vocab)}", f"Merges length: {len(merges)}")
