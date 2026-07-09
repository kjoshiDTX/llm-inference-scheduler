"""Week 1 exit criterion: our BPE matches the `tokenizers` reference exactly
on a diverse set of strings, and the from-scratch trainable BPE round-trips."""

import pytest

from mini_engine.tokenizer.bpe import ByteBPE
from mini_engine.tokenizer.qwen_tokenizer import (
    IncrementalDetokenizer,
    QwenTokenizer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TEST_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "   leading and trailing spaces   ",
    "tabs\tand\nnewlines\r\nand\r\n\r\nmore",
    "1234567890 123 4567",
    "CamelCaseAndSNAKE_CASE and kebab-case",
    "안녕하세요 世界 مرحبا Привет",
    "emoji: 🚀🔥💯 combined: 👨‍👩‍👧‍👦",
    "don't we're I'll it's THEY'RE",
    "http://example.com/path?query=1&x=2#frag",
    "price: $1,234.56 (–7.8%) ≈ €1.1k",
    "",
    " ",
    "a",
    "\n",
    "ückenschmerzen Straße naïve façade",
    "x" * 500,
    "word " * 100,
    "混合text with日本語and English",
]


@pytest.fixture(scope="module")
def tokenizers_pair():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    import os

    path = snapshot_download(MODEL_ID)
    ours = QwenTokenizer(os.path.join(path, "tokenizer.json"))
    ref = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
    return ours, ref


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_encode_matches_reference(tokenizers_pair, text):
    ours, ref = tokenizers_pair
    assert ours.encode(text) == ref.encode(text).ids, f"mismatch on {text!r}"


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_roundtrip(tokenizers_pair, text):
    ours, _ = tokenizers_pair
    assert ours.decode(ours.encode(text)) == text


def test_chat_template_matches_hf(tokenizers_pair):
    ours, _ = tokenizers_pair
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a KV cache?"},
    ]
    expected = hf.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    assert ours.apply_chat_template(messages) == expected


def test_incremental_detokenizer_never_breaks_utf8(tokenizers_pair):
    """Multi-byte codepoints span token boundaries; the streaming path must
    buffer partial bytes instead of emitting U+FFFD."""
    ours, _ = tokenizers_pair
    text = "🚀 火箭 roket ракета 로켓" * 3
    ids = ours.encode(text)
    detok = IncrementalDetokenizer(ours)
    streamed = "".join(detok.push(i) for i in ids) + detok.flush()
    assert streamed == text
    assert "�" not in streamed


def test_from_scratch_bpe_trains_and_roundtrips():
    corpus = "the cat sat on the mat and the dog sat on the log " * 50
    bpe = ByteBPE()
    bpe.train(corpus, vocab_size=300)
    assert len(bpe.merges) > 0
    for text in ["the cat", "unseen words entirely", "ünïcödé"]:
        assert bpe.decode(bpe.encode(text)) == text
    # Frequent words should compress below byte length.
    assert len(bpe.encode("the cat sat on the mat")) < len("the cat sat on the mat".encode())
