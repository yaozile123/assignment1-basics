import os
import regex as re
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

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


def remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Pre-tokenization step for BPE: split text by special tokens while preserving them.
    
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
