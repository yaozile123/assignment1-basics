import io

from cs336_basics.utils import remove_special_tokens, split_chunks


def test_remove_special_tokens_basic():
    """Test basic splitting with a single special token."""
    text = "Hello world<|endoftext|>This is a test<|endoftext|>End"
    special_tokens = ["<|endoftext|>"]
    expected = ["Hello world", "This is a test", "End"]
    result = remove_special_tokens(text, special_tokens)
    assert result == expected


def test_remove_special_tokens_multiple_tokens():
    """Test splitting with multiple different special tokens."""
    text = "Start<|begin|>middle<|end|>finish"
    special_tokens = ["<|begin|>", "<|end|>"]
    expected = ["Start", "middle", "finish"]
    result = remove_special_tokens(text, special_tokens)
    assert result == expected


def test_remove_special_tokens_edge_cases():
    """Test edge cases: empty text, no special tokens found, regex characters."""
    # Empty text
    text = ""
    special_tokens = ["<|endoftext|>"]
    expected = []
    result = remove_special_tokens(text, special_tokens)
    assert result == expected

    # No special tokens in text
    text = "No special tokens here"
    special_tokens = ["<|endoftext|>"]
    expected = ["No special tokens here"]
    result = remove_special_tokens(text, special_tokens)
    assert result == expected

    # Special tokens with regex characters
    text = "Hello (test) world + question ?"
    special_tokens = ["(test)", "+", "?"]
    expected = ["Hello ", " world ", " question "]
    result = remove_special_tokens(text, special_tokens)
    assert result == expected


def test_split_chunks_basic():
    """Test basic chunk splitting with multiple boundaries."""
    data = b"0123456789ABCDEFGHIJ"
    file = io.BytesIO(data)
    boundaries = [0, 5, 10, 15, len(data)]

    chunks = split_chunks(file, boundaries)

    assert len(chunks) == 4
    assert chunks[0] == "01234"
    assert chunks[1] == "56789"
    assert chunks[2] == "ABCDE"
    assert chunks[3] == "FGHIJ"


def test_split_chunks_single_boundary():
    """Test chunk splitting with a single boundary."""
    data = b"HelloWorld"
    file = io.BytesIO(data)
    boundaries = [len(data)]

    chunks = split_chunks(file, boundaries)

    assert len(chunks) == 1
    assert chunks[0] == "HelloWorld"
