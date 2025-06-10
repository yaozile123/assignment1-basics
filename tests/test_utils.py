import io

from cs336_basics.utils import pretokenize_chunk, remove_special_tokens, split_chunks


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


def test_pretokenize_chunk_basic():
    """Test basic tokenization with simple text."""
    chunk = "hello world"
    special_tokens = []
    result = pretokenize_chunk(chunk, special_tokens)
    assert result["hello"] == 1
    assert result[" world"] == 1


def test_pretokenize_chunk_with_special_tokens():
    """Test tokenization with special tokens present."""
    chunk = "hello<|endoftext|>world"
    special_tokens = ["<|endoftext|>"]
    result = pretokenize_chunk(chunk, special_tokens)
    assert result["hello"] == 1
    assert result["world"] == 1


def test_pretokenize_chunk_with_multiple_special_tokens():
    """Test tokenization with multiple special tokens present."""
    chunk = "hello<|endoftext|>world<|PAD|>!"
    special_tokens = ["<|endoftext|>", "<|PAD|>"]
    result = pretokenize_chunk(chunk, special_tokens)
    assert result["hello"] == 1
    assert result["world"] == 1
    assert result["!"] == 1


def test_pretokenize_chunk_empty():
    """Test tokenization with empty inputs."""
    result1 = pretokenize_chunk("", [])
    assert len(result1) == 0

    result2 = pretokenize_chunk("", ["<|endoftext|>"])
    assert len(result2) == 0


def test_pretokenize_chunk_mixed_content():
    """Test tokenization with numbers and special characters."""
    chunk = "123 hello !@# 456 world"
    special_tokens = ["!", "@", "#"]
    result = pretokenize_chunk(chunk, special_tokens)
    assert result["123"] == 1
    assert result[" 456"] == 1
    assert result[" hello"] == 1
    assert result[" world"] == 1
    assert result[" "] == 1
    assert len(result) == 5
