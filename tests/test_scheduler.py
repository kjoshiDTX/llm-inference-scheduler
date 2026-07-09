"""Week 8 exit criteria, with the real model (CPU fp32, short generations):

1. Continuous batching + paged KV + prefix cache must produce EXACTLY the
   greedy tokens of the simple single-sequence path — the entire memory
   system is an optimization, not a behavior change.
2. Prefix cache actually skips work on shared prefixes.
3. Preemption under a tiny block pool still completes correctly.
"""

import pytest
import torch

from mini_engine.engine import Engine
from mini_engine.generate import generate
from mini_engine.model.loader import load_weights
from mini_engine.model.qwen2 import Qwen2Model
from mini_engine.sampling import SamplingParams
from mini_engine.tokenizer.qwen_tokenizer import QwenTokenizer

MAX_TOKENS = 24
PROMPTS = [
    "The three most important ideas in operating systems are",
    "Write a haiku about memory fragmentation.",
    "Explain TCP slow start briefly:",
]
SHARED_SYSTEM = (
    "You are a meticulous systems engineer who answers concisely and "
    "always cites the mechanism, not the folklore. "
)


@pytest.fixture(scope="module")
def engine():
    eng = Engine(num_blocks=512, max_num_seqs=4, max_batched_tokens=128)
    return eng


@pytest.fixture(scope="module")
def reference(engine):
    """Greedy reference outputs via the simple Week-4 path (same weights)."""
    model = engine.model
    tok = engine.tokenizer
    out = {}
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}])
        params = SamplingParams(temperature=0.0, max_new_tokens=MAX_TOKENS)
        out[p] = generate(model, tok, ids, params).token_ids
    return out


def test_concurrent_batched_matches_single_sequence(engine, reference):
    reqs = []
    for p in PROMPTS:
        ids = engine.tokenizer.apply_chat_template([{"role": "user", "content": p}])
        reqs.append(
            engine.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=MAX_TOKENS))
        )
    # Drive the loop synchronously until all finish.
    while engine.scheduler.has_work:
        engine.scheduler.step()
    for p, req in zip(PROMPTS, reqs):
        assert req.output_ids == reference[p], (
            f"batched output diverged for {p!r}:\n{req.output_ids}\nvs\n{reference[p]}"
        )


def test_prefix_cache_hits_on_shared_system_prompt(engine):
    msgs = lambda q: [
        {"role": "system", "content": SHARED_SYSTEM},
        {"role": "user", "content": q},
    ]
    params = lambda: SamplingParams(temperature=0.0, max_new_tokens=8)

    r1 = engine.submit(engine.tokenizer.apply_chat_template(msgs("What is paging?")), params())
    while engine.scheduler.has_work:
        engine.scheduler.step()
    hits_before = engine.scheduler.metrics["prefix_cache_hit_tokens"]

    r2 = engine.submit(engine.tokenizer.apply_chat_template(msgs("What is a TLB?")), params())
    while engine.scheduler.has_work:
        engine.scheduler.step()
    hits_after = engine.scheduler.metrics["prefix_cache_hit_tokens"]

    shared_len = len(
        engine.tokenizer.encode(f"<|im_start|>system\n{SHARED_SYSTEM}", allow_special=True)
    )
    assert hits_after - hits_before >= (shared_len // 16 - 1) * 16, (
        "second request should reuse the shared system prompt's blocks"
    )
    assert r1.finish_reason and r2.finish_reason


def test_prefix_cached_output_identical(engine, reference):
    """Reusing cached prefix blocks must not change greedy outputs."""
    p = PROMPTS[0]
    ids = engine.tokenizer.apply_chat_template([{"role": "user", "content": p}])
    req = engine.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=MAX_TOKENS))
    while engine.scheduler.has_work:
        engine.scheduler.step()
    assert req.num_prefix_cached > 0, "identical earlier prompt should be cached by now"
    assert req.output_ids == reference[p]


def test_preemption_under_memory_pressure():
    """A pool of 11 blocks (176 tokens) with 3 concurrent sequences growing
    to ~86 tokens (6 blocks) forces preemption; all must still finish
    correctly (recompute restores exact state)."""
    eng = Engine(num_blocks=11, max_num_seqs=3, max_batched_tokens=64,
                 enable_prefix_cache=False)
    reqs = []
    for p in PROMPTS:
        ids = eng.tokenizer.apply_chat_template([{"role": "user", "content": p}])
        reqs.append(eng.submit(ids, SamplingParams(temperature=0.0, max_new_tokens=48)))
    steps = 0
    while eng.scheduler.has_work and steps < 5000:
        eng.scheduler.step()
        steps += 1
    assert all(r.finish_reason in ("stop", "length") for r in reqs)
    assert eng.scheduler.metrics["preemptions"] > 0, (
        "test is mis-sized: no preemption happened"
    )
    # Reference check for one of them on a fresh unconstrained engine path.
    ref = generate(
        eng.model, eng.tokenizer,
        eng.tokenizer.apply_chat_template([{"role": "user", "content": PROMPTS[0]}]),
        SamplingParams(temperature=0.0, max_new_tokens=48),
    ).token_ids
    assert reqs[0].output_ids == ref
