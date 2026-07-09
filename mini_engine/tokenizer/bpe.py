"""Week 1: byte-level BPE from first principles.

Problem: models eat integers, users send UTF-8. Char-level tokenization makes
sequences ~4x longer (4x the KV cache); word-level has unbounded vocab. BPE
starts from 256 byte tokens and greedily merges the most frequent adjacent
pair until the vocab budget is spent — frequent strings become single tokens,
anything (any bytes at all) remains representable.

This module is the trainable from-scratch version used by the tests;
qwen_tokenizer.py loads Qwen's real merges with the same merge loop.
"""

from collections import Counter


def get_pair_counts(sequences: list[list[int]]) -> Counter:
    counts: Counter = Counter()
    for seq in sequences:
        for pair in zip(seq, seq[1:]):
            counts[pair] += 1
    return counts


def merge_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class ByteBPE:
    """Minimal byte-level BPE: train on a corpus, encode/decode any bytes."""

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # pair -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, corpus: str, vocab_size: int):
        assert vocab_size >= 256
        sequences = [list(chunk.encode("utf-8")) for chunk in corpus.split()]
        for new_id in range(256, vocab_size):
            counts = get_pair_counts(sequences)
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            sequences = [merge_pair(s, pair, new_id) for s in sequences]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # Apply merges in training order (= rank order): lowest-rank pair first.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidates = [(self.merges[p], p) for p in pairs if p in self.merges]
            if not candidates:
                break
            _, best = min(candidates)
            ids = merge_pair(ids, best, self.merges[best])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
