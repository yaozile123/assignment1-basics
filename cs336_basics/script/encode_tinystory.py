from cs336_basics.tokenizer import Tokenizer
from cs336_basics.preprocess_data import preprocess_text_data

tinystory_data = {
    'train':'data/TinyStoriesV2-GPT4-train.txt',
    'val':'data/TinyStoriesV2-GPT4-valid.txt',
    'vocab_filepath': 'data/out/tinystories_vocab.pkl',
    'merges_filepath': 'data/out/tinystories_merges.pkl',
    'special_tokens': ['<|endoftext|>']
}

for split in ['train', 'val']:
    tokenizer = Tokenizer.from_files(
        vocab_filepath=tinystory_data['vocab_filepath'],
        merges_filepath=tinystory_data['merges_filepath'],
        special_tokens=tinystory_data['special_tokens']
    )
    preprocess_text_data(
        text_file_path=tinystory_data[f'{split}'],
        output_path=f'data/out/tinystories_{split}_tokenized.npy',
        tokenizer=tokenizer
    )