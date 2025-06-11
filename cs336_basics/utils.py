import os
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_chunks(file: BinaryIO, chunk_boundaries: list[int]) -> list[str]:
    """
    Split the file into chunks based on the provided boundaries.

    Args:
        file: A binary file object to read from.
        chunk_boundaries: A list of byte offsets where the file should be split.

    Returns:
        A list of strings, each representing a chunk of the file.
    """
    if not chunk_boundaries:
        raise ValueError("Chunk boundaries list cannot be empty.")
    if len(chunk_boundaries) == 1:
        # If there's only one boundary, return the whole file as a single chunk
        file.seek(0)
        return [file.read().decode("utf-8", errors="ignore")]
    chunks = []
    for i in range(len(chunk_boundaries) - 1):
        start = chunk_boundaries[i]
        end = chunk_boundaries[i + 1]
        file.seek(start)
        chunks.append(file.read(end - start).decode("utf-8", errors="ignore"))
    return chunks


def remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Pre-tokenization step for BPE: split text by special tokens.

    Args:
        text: Input text to be pre-tokenized
        special_tokens: List of special tokens that should be preserved as single units

    Returns:
        List of text segments, where special tokens are preserved as separate elements
        and non-special text is split from them
    """
    if not special_tokens:
        return [text] if text else []

    # Create pattern that captures special tokens
    pattern = "|".join(re.escape(token) for token in special_tokens)

    # Split text and removing the special tokens
    parts = re.split(pattern, text)

    # Filter out empty strings and return non-empty parts
    return [part for part in parts if part]


def pretokenize_chunk(chunk: str, special_tokens: list[str]) -> list[str]:
    """
    Pre-tokenization step for a single chunk: split text by special tokens.

    Args:
        chunk: Input text chunk to be pre-tokenized
        special_tokens: List of special tokens that should be preserved as single units

    Returns:
        List of text segments, where special tokens are preserved as separate elements
        and non-special text is split from them
    """
    cleaned_chunks = remove_special_tokens(chunk, special_tokens)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counter = Counter()
    for chunk in cleaned_chunks:
        for matched_word in re.finditer(PAT, chunk):
            counter[matched_word.group()] += 1
    return counter


def get_pair_counts(counter: Counter) -> tuple[Counter, dict, dict]:
    """
    Get the most frequent pair of adjacent tokens in the counter.

    Args:
        counter: A Counter object containing word frequencies.

    Returns:
        pair_counts: A Counter object containing counts of adjacent token pairs.
        word_to_pairs: A defaultdict mapping words to their adjacent token pairs and their counts.
        pairs_to_word: A defaultdict mapping pairs to the words they appear in.
    """
    pair_counts = Counter()
    word_splits = {word: [bytes([b]) for b in word.encode('utf-8')] for word in counter}
    pairs_to_word = defaultdict(set)
    for word, count in counter.items():
        tokens = word_splits[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
            pairs_to_word[pair].add(word)
    return pair_counts, word_splits, pairs_to_word


def get_most_frequent_pair(pair_counts: Counter) -> tuple[bytes, bytes]:
    """
    Get the most frequent pair of adjacent tokens from the pair counts.
    If ties, return the largest pair lexicographically.

    Args:
        pair_counts: A Counter object containing token frequencies.

    Returns:
        The most frequent pair of adjacent tokens.
    """
    if not pair_counts:
        return None, None
    most_frequent = max(pair_counts.values())
    most_frequent_pairs = [pair for pair in pair_counts if pair_counts[pair] == most_frequent]
    return max(most_frequent_pairs)
