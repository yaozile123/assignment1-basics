## Design Doc for assignment 1

### 1. Tokenization
`bpe.py` is the main file for tokenization.
it contains the following functions:
- `pretokenization`: pretokenize the text
- `bpe_merge`: merge the tokens
- `train_bpe`: train the BPE tokenizer
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.
Your BPE training function should return the resulting vocabulary and merges:
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
lary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.


`utils.py` contains the following functions:
- `remove_special_tokens`: remove special tokens from the text
- `find_chunk_boundaries`: find the chunk boundaries of the text
