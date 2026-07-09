"""Week 4 milestone: the single-sequence generation loop (mini-engine v0.1).

Prefill the prompt in one forward, then decode token-by-token, streaming
through the incremental detokenizer. The batched/scheduled path lives in
scheduler.py; this loop exists for the CLI chat demo and the with/without-
cache benchmark.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from .kv_cache import ContiguousKVCache, NoCache
from .model.qwen2 import Qwen2Model, SequenceForwardInfo
from .sampling import SamplingParams, sample
from .tokenizer.qwen_tokenizer import IncrementalDetokenizer, QwenTokenizer


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    ttft_s: float
    itl_s: list[float] = field(default_factory=list)

    @property
    def decode_tps(self) -> float:
        total = sum(self.itl_s)
        return (len(self.itl_s) / total) if total > 0 else 0.0


@torch.inference_mode()
def generate(
    model: Qwen2Model,
    tokenizer: QwenTokenizer,
    prompt_ids: list[int],
    params: SamplingParams,
    on_token: Callable[[str], None] | None = None,
    use_cache: bool = True,
) -> GenerationResult:
    cfg = model.config
    if use_cache:
        cache = ContiguousKVCache(
            cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim,
            device=model.device, dtype=model.dtype,
        )
    else:
        cache = NoCache(cfg.num_hidden_layers)
    cache.add_sequence(0)

    stop_ids = params.stop_token_ids or tokenizer.eos_ids
    detok = IncrementalDetokenizer(tokenizer)
    gen = None
    if params.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(params.seed)

    all_ids = list(prompt_ids)
    out_ids: list[int] = []
    text_parts: list[str] = []
    itl: list[float] = []

    def step(feed_ids: list[int], start_pos: int) -> int:
        ids = torch.tensor(feed_ids, device=model.device)
        info = SequenceForwardInfo(
            seq_id=0, flat_start=0, num_tokens=len(feed_ids),
            positions=torch.arange(start_pos, start_pos + len(feed_ids)),
        )
        logits = model.forward(ids, [info], cache)[0]
        return sample(logits, params, out_ids, gen)

    t0 = time.perf_counter()
    next_id = step(all_ids, 0)  # prefill
    ttft = time.perf_counter() - t0

    last = t0 + ttft
    while True:
        out_ids.append(next_id)
        all_ids.append(next_id)
        if next_id not in stop_ids:
            piece = detok.push(next_id)
            if piece:
                text_parts.append(piece)
                if on_token:
                    on_token(piece)
        if next_id in stop_ids or len(out_ids) >= params.max_new_tokens:
            break
        if use_cache:
            next_id = step([next_id], len(all_ids) - 1)  # decode: one token
        else:
            # Naive baseline: re-run the entire sequence every step.
            next_id = step(all_ids, 0)
        now = time.perf_counter()
        itl.append(now - last)
        last = now

    text_parts.append(detok.flush())
    return GenerationResult(out_ids, "".join(text_parts), ttft, itl)
