"""Week 3's 'single most clarifying benchmark': tokens/sec with vs without
the KV cache at long context. Without the cache every decode step re-runs
the full forward over the whole sequence — O(n^2) per token.

Usage: python benchmarks/bench_kv_cache.py [--context 1000] [--decode 32]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch

from mini_engine.generate import generate
from mini_engine.model.loader import load_weights
from mini_engine.model.qwen2 import Qwen2Model
from mini_engine.sampling import SamplingParams
from mini_engine.tokenizer.qwen_tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--decode", type=int, default=32)
    args = ap.parse_args()

    config, weights = load_weights(device="cpu", dtype=torch.float32)
    model = Qwen2Model(config, weights)
    tok = QwenTokenizer.from_model()

    # Build a prompt of ~context tokens.
    base = tok.encode("The history of computing is long and winding. ")
    prompt = (base * (args.context // len(base) + 1))[: args.context]
    params = SamplingParams(temperature=0.0, max_new_tokens=args.decode)

    print(f"context={len(prompt)} tokens, decoding {args.decode} tokens\n")
    for use_cache in (True, False):
        t0 = time.perf_counter()
        result = generate(model, tok, prompt, params, use_cache=use_cache)
        wall = time.perf_counter() - t0
        label = "WITH cache   " if use_cache else "WITHOUT cache"
        print(
            f"{label}: ttft={result.ttft_s:6.2f}s  "
            f"decode={result.decode_tps:6.2f} tok/s  total={wall:6.2f}s"
        )

    kb = config.kv_bytes_per_token / 1024
    print(f"\nKV cache cost: {kb:.0f} KB/token fp16 "
          f"({len(prompt)} tokens = {kb * len(prompt) / 1024:.1f} MB)")
    print(f"24 GB GPU fits ~{int(24e9 / config.kv_bytes_per_token / 4096)} "
          f"concurrent 4k-token conversations (weights aside)")
    full_mha = config.kv_bytes_per_token * (
        config.num_attention_heads / config.num_key_value_heads
    )
    print(f"If Qwen used full MHA (14 KV heads): {full_mha/1024:.0f} KB/token "
          f"— GQA buys the 7x that makes cache-aware routing worth doing")


if __name__ == "__main__":
    main()
