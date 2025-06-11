import pickle
import regex as re
from typing import Iterable, Iterator

from cs336_basics.utils import create_new_tokens

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes] | None = None):
        self.id_to_token = vocab
        self.token_to_id = {token: index for index, token in vocab.items()}
        self.merges = {x: i for i, x in enumerate(merges)}
        self.special_tokens = set([token for token in special_tokens]) if special_tokens else set()
        for token in self.special_tokens:
            encoded_token = token.encode('utf-8')
            if encoded_token not in self.token_to_id:
                self.id_to_token[len(self.token_to_id)] = encoded_token
                self.token_to_id[encoded_token] = len(self.token_to_id)

    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[bytes] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        ids = []
        if not self.special_tokens:
            return self.encode_part(text)
        special_pattern = "|".join(
            re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        pattern = f"({special_pattern})"
        parts = re.split(pattern, text)
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.token_to_id[part.encode('utf-8')])
            else:
                ids.extend(self.encode_part(part))
        return ids
    

    def encode_part(self, part: str) -> list[int]:
        ids = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, part):
            bytes_sequence = match.group().encode("utf-8")
            ids.extend(self.encode_bytes(bytes_sequence))
        return ids


    def encode_bytes(self, bytes_sequence: bytes) -> list[int]:
        byte_list = [bytes([byte]) for byte in bytes_sequence]
        while len(byte_list) > 1:
            pairs = set([(byte_list[i], byte_list[i + 1]) for i in range(len(byte_list) - 1)])
            # find pair with lowest index in merges
            pair = min(pairs, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            byte_list = create_new_tokens(byte_list, pair)
        return [self.token_to_id[byte] for byte in byte_list]


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[id_] for id_ in ids]
        byte_data = b''.join(tokens)
        return byte_data.decode('utf-8', errors='replace')