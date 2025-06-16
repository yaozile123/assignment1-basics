import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import Tokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def preprocess_text_data(
    text_file_path: str, 
    output_path: str, 
    tokenizer: Tokenizer
):
    """
    Preprocess raw text data into tokenized format for training.
    
    Args:
        text_file_path (str): Path to the raw text file
        output_path (str): Path to save the tokenized data (.npy file)
        tokenizer (Tokenizer): The tokenizer to use for encoding
    """
    logging.info(f"Preprocessing text data from {text_file_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    token_ids = []
    for line in text_iterator(text_file_path):
        token_ids.extend(tokenizer.encode(line))
    total_tokens = len(token_ids)
    logging.info(f"Tokenization complete. Total tokens: {total_tokens:,}")
    token_array = np.array(token_ids, dtype=np.uint16)
    # Save as numpy array
    np.save(output_path, token_array)
    logging.info(f"Saved {total_tokens:,} tokens to {output_path}")
    logging.info(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return total_tokens


def text_iterator(file_path):
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    print("Total lines: ", total_lines)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc="Processing"):
            yield line.strip()

    