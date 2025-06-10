from collections import Counter

from cs336_basics.bpe import (
    bpe_merge,
    initialize_vocab,
    paralleize_pretokenization,
    train_bpe,
)


def test_initialize_vocab():
    special_tokens = ["<pad>", "<unk>"]
    vocab = initialize_vocab(special_tokens)
    assert len(vocab) == 258  # 256 byte tokens + 2 special tokens
    assert vocab[256] == b"<pad>"
    assert vocab[257] == b"<unk>"


def test_paralleize_pretokenization_basic():
    chunks = [
        "Hello, world! This is a test.",
        "Testing special tokens: <pad> and <unk> should be preserved.",
    ]
    special_tokens = ["<pad>", "<unk>"]

    # Use only 2 processes for a lightweight test
    counter = paralleize_pretokenization(
        chunks, num_of_processes=2, special_tokens=special_tokens
    )

    # Check that it returns a Counter
    assert isinstance(counter, Counter)

    # Check that some expected tokens exist
    assert counter["Hello"] == 1
    assert counter[" world"] == 1
    assert counter["."] == 2
    assert counter["<pad>"] == 0
    assert counter["<unk>"] == 0


def test_bpe_merge_basic():
    """Test basic BPE merge operation with simple words"""
    print("Testing simple merge...")

    # Input: "hello" appears 2 times, "hell" appears 1 time
    counter = Counter({"hello": 2, "hell": 1})

    # Initial vocab: 256 single-byte tokens
    vocab = {i: bytes([i]) for i in range(256)}

    # Perform 1 merge
    new_vocab, merges = bpe_merge(counter, vocab, 1)

    # Verify results
    assert len(merges) == 1, f"Expected 1 merge, got {len(merges)}"

    # Most frequent pair should be (b'l', b'l') appearing 3 times total
    expected_pair = (b"l", b"l")
    assert merges[0] == expected_pair, f"Expected {expected_pair}, got {merges[0]}"

    # Vocab should grow by one token
    assert len(new_vocab) == 257, f"Expected vocab size 257, got {len(new_vocab)}"
    assert b"ll" in new_vocab.values(), "Merged token 'll' not found in vocab"
    assert new_vocab[256] == b"ll", "Merged token 'll' should be at index 256"


def test_bpe_merge_multiple():
    """Test consecutive merge operations"""
    counter = Counter({"aaa": 3})
    vocab = {i: bytes([i]) for i in range(256)}

    new_vocab, merges = bpe_merge(counter, vocab, 2)

    # Should perform exactly 2 merges
    assert len(merges) == 2, f"Expected 2 merges, got {len(merges)}"

    # First merge should combine adjacent 'a' characters
    assert merges[0] == (
        b"a",
        b"a",
    ), f"First merge should be (b'a', b'a'), got {merges[0]}"

    # Vocab should grow by 2 tokens
    assert len(new_vocab) == 258, f"Expected vocab size 258, got {len(new_vocab)}"


def test_bpe_merge_complex_patterns():
    """Test merge operations with overlapping and competing patterns"""
    counter = Counter(
        {
            "abcabc": 4,  # Contains patterns: ab, bc, ca, ab, bc
            "bcabca": 3,  # Contains patterns: bc, ca, ab, bc, ca
            "ababab": 2,  # Contains patterns: ab, ba, ab, ba, ab
        }
    )
    vocab = {i: bytes([i]) for i in range(256)}

    new_vocab, merges = bpe_merge(counter, vocab, 3)

    # Should perform exactly 3 merges
    assert len(merges) == 3, f"Expected 3 merges, got {len(merges)}"

    # Calculate expected frequencies:
    # (b'a', b'b'): appears in "abcabc"*4 (2 times) + "bcabca"*3 (1 time) + "ababab"*2 (3 times) = 8+3+6 = 17 total
    # (b'b', b'c'): appears in "abcabc"*4 (2 times) + "bcabca"*3 (2 times) = 8+6 = 14 total
    # (b'c', b'a'): appears in "abcabc"*4 (1 time) + "bcabca"*3 (2 times) = 4+6 = 10 total
    # (b'b', b'a'): appears in "ababab"*2 (2 times) = 4 total

    # First merge should be most frequent pair (a,b) with count 17
    assert merges[0] == (
        b"a",
        b"b",
    ), f"First merge should be (b'a', b'b'), got {merges[0]}"

    # After first merge, new patterns emerge with the merged token
    # Verify vocab growth
    assert len(new_vocab) == 259, f"Expected vocab size 259, got {len(new_vocab)}"
    assert b"ab" in new_vocab.values(), "Merged token 'ab' should be in vocab"
