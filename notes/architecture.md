# Anatomy of this inference system (Week 0 → Week 14, after the fact)

## Life of a request

```
client
  │  POST /v1/chat/completions {messages, stream:true}
  ▼
Go router (:8000)                        router/main.go
  │  policy.Pick(): score = α·prefix_overlap(chars) − β·inflight
  │  per-worker approximate radix tree of forwarded prefixes
  ▼
worker (:8001 / :8002)                   server/worker.py
  │  FastAPI handler → Engine.submit_chat()
  ▼
tokenizer                                mini_engine/tokenizer/
  │  chat template → token IDs (byte-level BPE, ours, matches `tokenizers`)
  ▼
scheduler (one iteration = one step)     mini_engine/scheduler.py
  │  admit if blocks free ── radix.match_prefix() skips cached blocks
  │  decode-first token budget (chunked prefill) ── preempt on OOM (recompute)
  ▼
block manager / radix cache              mini_engine/paged/
  │  16-token blocks, ref counts, COW, LRU eviction pinned by ref counts
  ▼
model executor                           mini_engine/model/qwen2.py
  │  flat token batch → RMSNorm → RoPE → GQA attention (gather K/V through
  │  block tables) → SwiGLU ×24 → logits for each sequence's last token
  ▼
sampler                                  mini_engine/sampling.py
  │  rep-penalty → temperature → top-k → top-p → multinomial
  ▼
incremental detokenizer                  (UTF-8 boundary buffering)
  │
  ▼ SSE chunks ── back through router (flush per chunk) ── client
```

Cancellation path: client disconnect → router's request context cancels the
upstream call → worker's SSE generator teardown → `engine.cancel()` → blocks
decref'd, memory back in the free list within one step.

## The KV arithmetic that drives every design choice

Qwen2.5-0.5B, GQA 14Q/2KV, head_dim 64, 24 layers:

| quantity | value |
|---|---|
| KV bytes/token (fp16) | 2 × 24 × 2 × 64 × 2 = **12 KB** |
| same, with full MHA (14 KV heads) | 84 KB — GQA's 7× is why routing for cache locality is worth anything |
| 4k-token conversation | 48 MB |
| conversations in 24 GB (weights aside) | ~476 |
| block (16 tokens) | 192 KB fp16 |

Scarce resource → sharing (paged blocks + radix tree) → placement policy
(router). The whole stack exists because of the first row of this table.

## What each week's "naive fail" was

| layer | naive approach | failure | fix (here) |
|---|---|---|---|
| tokenizer | char-level | 4× sequence length = 4× KV | byte-level BPE |
| generation | recompute per token | O(n²)/token; 1.9 vs 18.9 tok/s at ctx 600 | KV cache |
| batching | static batches | convoy: batch ends with its longest member | continuous batching |
| memory | contiguous per-request | fragmentation, zero sharing | paged blocks |
| prefix reuse | none | system prompts re-prefilled per request | radix cache |
| routing | round-robin | ignores which worker holds the KV | overlap − queue-depth scoring |
| overload | admit everything | preemption thrash | admission on free blocks |

## Simplifications vs production (known, deliberate)

- Attention gathers K/V per sequence in Python; vLLM fuses gather+attention
  in a paged CUDA kernel (FlashInfer et al.). Mechanism identical, speed not.
- Radix tree edges are block-aligned; SGLang splits at arbitrary tokens.
- Preemption implements recompute only; swap-to-host is explained in
  scheduler.py's docstring but not implemented.
- Router's prefix tree is character-level and approximate; sgl-router keeps
  token-accurate state fed back from workers.
- No tensor/pipeline parallelism; 0.5B fits one device. Prefill/decode
  disaggregation (DistServe/Mooncake) is future work.
- fp32 on CPU because the benchmark cluster is a MacBook; the memory math in
  the README uses fp16 numbers because that's what a GPU deployment would use.
