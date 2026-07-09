# llm-inference-scheduler

A distributed LLM inference system where **every process is code in this repo**:
a from-scratch inference engine (tokenizer → transformer forward pass → paged
KV cache → continuous-batching scheduler), an OpenAI-compatible worker server,
a cache-aware Go router, and an open-loop load generator — benchmarked
end-to-end against round-robin and hash-affinity routing.

No `model.generate()`, no `pipeline()`, no vLLM. HuggingFace is used to
download weights and to *diff against* in tests, never to run anything.

```
loadgen (Python, open-loop Poisson)
   └─> router (Go): score(worker) = α·prefix_overlap − β·queue_depth
         ├─> worker 1 (FastAPI + mini_engine)
         └─> worker 2 (FastAPI + mini_engine)
              └─ continuous batching · paged KV (16-token blocks, ref counts,
                 COW) · radix prefix cache (LRU + pinning) · chunked prefill ·
                 recompute preemption · GQA attention · streaming detokenizer
```

Model: Qwen2.5-0.5B-Instruct (24 layers, hidden 896, 14 query / 2 KV heads).
The GQA layout is the whole story in one number: **12 KB of KV per token**
(fp16) instead of MHA's 84 KB — cache is the scarce resource, so who holds
which prefix decides your TTFT. See [notes/architecture.md](notes/architecture.md).

## Results

**KV cache (the reason the engine exists).** Decode at context 600, CPU fp32:
18.9 tok/s with cache vs 1.9 tok/s recomputing — the O(n²)-per-token tax,
measured. (`benchmarks/bench_kv_cache.py`)

**Prefix cache + cache-aware routing (the reason the router exists).**
Multi-turn conversation, turn 2: 64/87 prompt tokens skipped. New conversation
sharing only the system prompt: 48 tokens skipped — the case conversation-ID
hashing is structurally blind to. (integration test, Week 12)

**Headline chart** — p99 TTFT and goodput vs offered load, three routing
policies, two workers, multi-turn + shared-system-prompt workload:

![headline](benchmarks/results/headline.png)

## Correctness discipline

The memory system must be a pure optimization. Tests enforce it:

- tokenizer matches `tokenizers` **exactly** on 20 adversarial strings
  (`test_tokenizer.py`)
- forward pass matches HF logits to <2e-3 fp32 (`test_logits_match_hf.py`)
- KV-cached decode ≡ full recompute (`test_logits_match_hf.py`)
- continuous-batched + paged + prefix-cached + **preempted** greedy outputs ≡
  single-sequence reference, token for token (`test_scheduler.py`)
- COW isolates forked sequences; LRU eviction respects ref-count pinning
  (`test_block_manager.py`, `test_radix_cache.py`)

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd router && go build -o router .)

.venv/bin/python -m pytest tests -q                 # ~2 min, downloads 1 GB model once
.venv/bin/python scripts/chat.py                    # talk to your own engine
.venv/bin/python benchmarks/bench_kv_cache.py       # the O(n²) demo
.venv/bin/python benchmarks/run_benchmark.py        # the headline chart (~15 min)
```

Single worker + router by hand:

```bash
WORKER_ID=w1 .venv/bin/python -m uvicorn server.worker:app --port 8001 &
./router/router -policy cache_aware -workers http://localhost:8001 &
curl -N localhost:8000/v1/chat/completions -d '{
  "messages":[{"role":"user","content":"Why is decode bandwidth-bound?"}],
  "stream":true}'
```

## Layout

| path | what | curriculum week |
|---|---|---|
| `mini_engine/tokenizer/` | byte-level BPE from scratch + Qwen vocab loader, incremental UTF-8-safe detokenizer | 1 |
| `mini_engine/model/` | Qwen2 forward pass from safetensors: RMSNorm, RoPE, GQA, SwiGLU | 2 |
| `mini_engine/kv_cache.py` | contiguous KV cache + the no-cache baseline | 3 |
| `mini_engine/sampling.py`, `generate.py` | sampler chain, streaming generation loop | 4 |
| `mini_engine/paged/block_manager.py` | 16-token blocks, free list, ref counts, copy-on-write | 6 |
| `mini_engine/paged/radix_cache.py` | radix-tree prefix cache, LRU + pinning | 7 |
| `mini_engine/scheduler.py` | continuous batching, chunked prefill, recompute preemption | 8 |
| `mini_engine/quant.py` | group-wise absmax int8 (weight-only) | 10 |
| `server/worker.py` | OpenAI-shaped API, SSE, cancellation→block-free, /metrics | 11 |
| `router/` | Go streaming proxy; round-robin / hash / cache-aware policies | 12–13 |
| `loadgen/`, `benchmarks/` | open-loop Poisson load, policy sweep, headline chart | 14 |

## Honest limitations

Python-speed attention (per-sequence gather, no fused paged kernel); CPU fp32
because the cluster is a MacBook; block-aligned radix edges (SGLang splits
anywhere); recompute-only preemption (no swap); char-level approximate router
state (sgl-router is token-accurate); no tensor parallelism. Future work, in
the order it would matter: fp8 KV quantization (2× conversations per GPU),
worker→router cache-state feedback, live request migration (Llumnix),
prefill/decode disaggregation (DistServe, Mooncake).

## Papers this implements (in reading order)

BPE (Sennrich '16) · Attention Is All You Need §3 · Qwen2 report ·
MQA (Shazeer '19) · GQA (Ainslie '23) · **PagedAttention (Kwon '23)** ·
**RadixAttention/SGLang (Zheng '23)** · **Orca (Yu, OSDI '22)** ·
Sarathi-Serve (chunked prefill) · FlashAttention (conceptual, see notes) ·
AWQ/GPTQ (motivation for quant.py) · Llumnix / DistServe / Mooncake (future work)
