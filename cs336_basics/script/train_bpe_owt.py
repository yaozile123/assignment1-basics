import time
import pickle
from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    file_path = "data/owt_train.txt"
    vocab_size = 32000
    start_time = time.time()
    vocab, merges = train_bpe(file_path, vocab_size, ["<|endoftext|>"])
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Vocab size: {len(vocab)}")
    print(f"Merges: {len(merges)}")
    longest_vocab = sorted(vocab.values(), key=len, reverse=True)[0]
    print(f"Longest Vocab has {len(longest_vocab)} bytes, which is {longest_vocab.decode('utf-8')}")
    
    # Create output directory if it doesn't exist
    os.makedirs("data/out", exist_ok=True)
    
    with open("data/out/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("data/out/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)