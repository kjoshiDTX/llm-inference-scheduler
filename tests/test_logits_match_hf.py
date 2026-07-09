"""Week 2's exit criterion: our from-scratch forward pass must match the
HF transformers reference to ~1e-3 (fp32, CPU). If this passes, every
downstream behavior difference is a bug in *our scheduler/cache*, never
ambiguity about the model math.
"""

import pytest
import torch

from mini_engine.kv_cache import ContiguousKVCache
from mini_engine.model.loader import MODEL_ID, load_weights
from mini_engine.model.qwen2 import Qwen2Model, SequenceForwardInfo
from mini_engine.tokenizer.qwen_tokenizer import QwenTokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "안녕하세요! Émojis: 🚀🔥. Mixed scripts test.",
]


@pytest.fixture(scope="module")
def setup():
    config, weights = load_weights(device="cpu", dtype=torch.float32)
    model = Qwen2Model(config, weights)
    tok = QwenTokenizer.from_model(MODEL_ID)

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    hf.eval()
    return model, tok, hf


@torch.inference_mode()
@pytest.mark.parametrize("prompt", PROMPTS)
def test_full_sequence_logits_match(setup, prompt):
    model, tok, hf = setup
    ids = tok.encode(prompt)
    cfg = model.config

    cache = ContiguousKVCache(
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    )
    cache.add_sequence(0)
    info = SequenceForwardInfo(0, 0, len(ids), torch.arange(len(ids)))
    ours = model.forward_all_logits(torch.tensor(ids), [info], cache)

    theirs = hf(torch.tensor([ids])).logits[0]

    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < 2e-3, f"max logit diff {max_diff}"


@torch.inference_mode()
def test_decode_with_cache_matches_full_forward(setup):
    """Token 501's logits must be identical whether computed incrementally
    through the KV cache or by re-running the whole sequence — this is the
    correctness contract that makes the cache an optimization, not a change."""
    model, tok, _ = setup
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 8)
    cfg = model.config

    # Incremental: prefill all but last token, then decode the last.
    cache = ContiguousKVCache(
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    )
    cache.add_sequence(0)
    n = len(ids)
    info = SequenceForwardInfo(0, 0, n - 1, torch.arange(n - 1))
    model.forward(torch.tensor(ids[: n - 1]), [info], cache)
    info = SequenceForwardInfo(0, 0, 1, torch.arange(n - 1, n))
    incremental = model.forward(torch.tensor(ids[n - 1 :]), [info], cache)[0]

    # Full recompute.
    cache2 = ContiguousKVCache(
        cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    )
    cache2.add_sequence(0)
    info = SequenceForwardInfo(0, 0, n, torch.arange(n))
    full = model.forward(torch.tensor(ids), [info], cache2)[0]

    assert (incremental - full).abs().max().item() < 1e-4
