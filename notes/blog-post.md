# I replaced vLLM with an engine I wrote from scratch — here's what my router taught me about KV caches

*Draft — publish after the final benchmark numbers land.*

Last spring I built a cache-aware router for LLM inference: hash the
conversation ID, pin the conversation to a worker, watch TTFT drop because
the worker already "had the conversation cached." It worked. I measured it.
I could not, however, have told you what "had it cached" physically meant —
which bytes, where, allocated by whom, evicted when. The router treated the
inference engine as a black box that rewarded locality.

This project erases that black box. I rebuilt the entire path — tokenizer,
transformer forward pass, KV cache, paged allocator, prefix cache, scheduler,
server — and then put my router back in front of it. Every process in the
final benchmark, from load generator to router to engine to block allocator,
is code I wrote.

## The number that explains the whole system

For Qwen2.5-0.5B with grouped-query attention, one token of KV cache costs

    2 (K and V) × 24 layers × 2 KV heads × 64 dims × 2 bytes = 12 KB.

Twelve kilobytes per token, forever, for the life of the conversation. A
4k-token conversation is 48 MB. A 24 GB GPU holds ~476 of them — *if* nothing
is wasted. That's the entire justification for three layers of machinery:

1. **Paged allocation** (vLLM's insight): contiguous per-request buffers
   fragment; 16-token blocks with ref counts don't. My OS course's virtual
   memory unit, load-bearing at last.
2. **Prefix caching** (SGLang's insight): identical prefixes — system prompts,
   multi-turn history — are identical KV. A radix tree over token IDs makes
   the shared blocks findable, ref counts keep eviction honest.
3. **Cache-aware routing** (my old project, now with mechanism): if the KV
   for a prefix lives on worker 1, sending the next turn to worker 2 pays
   full prefill. score = α·overlap − β·queue_depth.

My old router hashed conversation IDs because it couldn't see tokens. The
rebuilt one keeps an approximate radix tree per worker — and the benchmark
now shows the case hashing is structurally blind to: two *different*
conversations sharing a system prompt. Hash affinity scatters them; overlap
scoring stacks them on the worker that already computed those blocks. In my
integration test the second conversation skipped 48 tokens of prefill on its
very first request.

## Three things that surprised me

**The KV cache is not an optimization; it's the definition of decode.**
Without it, generating token n redoes attention *and the QKV projections* for
all n−1 predecessors: 1.9 tok/s vs 18.9 tok/s at context 600, on my laptop.
"Prefill fills the cache in one batch; decode appends one row" fell out of
the code before I read it anywhere.

**Streaming is a bytes problem, not a tokens problem.** BPE tokens end
mid-codepoint. Stream `decode(token)` per token and you emit U+FFFD garbage
on the first emoji. The fix — buffer bytes until they form valid UTF-8 — is
four lines, and exactly what TGI's incremental detokenizer does. I hit it
with a test string before I hit it in production; the curriculum called this
shot in Week 1.

**Preemption is just recompute wearing a scary name.** When the block pool
runs dry mid-generation, free the newest request's blocks and re-prefill it
later — prompt *plus* the tokens it already generated. My test proves the
preempted request's final output is token-identical to an unconstrained run.
Once you've written that, Orca's iteration-level scheduling section reads
like documentation of your own code.

## What I'd tell someone routing to a black box

Your router's policy is a bet about the engine's memory system. Mine bet on
conversation affinity and won small; it couldn't bet on cross-conversation
sharing because it didn't know sharing existed. You don't need to write a
whole engine to route well — but you do need to know what a block table is,
what pins it, and what evicts it, because α and β are denominated in those
units.

(Benchmark chart and final numbers: see the repo README.)
