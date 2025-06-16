from cs336_basics.tokenizer import Tokenizer
from cs336_basics.preprocess_data import preprocess_text_data

owt_data = {
    'train':'data/owt_train.txt',
    'val':'data/owt_val.txt',
    'vocab_filepath': 'data/out/owt_vocab.json',
    'merges_filepath': 'data/out/owt_merges.txt',
    'special_tokens': ['<|endoftext|>']
}

for split in ['train', 'val']:
    tokenizer = Tokenizer.from_files(
        vocab_filepath=owt_data['vocab_filepath'],
        merges_filepath=owt_data['merges_filepath'],
        special_tokens=owt_data['special_tokens']
    )
    preprocess_text_data(
        text_file_path=owt_data[f'{split}'],
        output_path=f'data/out/owt_{split}_tokenized.npy',
        tokenizer=tokenizer
    )