"""Open-loop load generator (the Week-14 workload).

Conversations arrive by a Poisson process at --rate conv/s regardless of
system backpressure (open loop — the load doesn't slow down because the
system is struggling; that's what makes saturation visible). Each
conversation runs --turns sequential turns; every turn resends the full
history, which is exactly the redundant prefill that prefix caching and
cache-aware routing exist to eliminate. Conversations share one of
--num-system-prompts long system prompts (cross-conversation reuse).

Per request: TTFT (first SSE content chunk), ITL between chunks, wall
latency, routed worker. Output: JSON records + a summary line.

Usage:
  python loadgen/loadgen.py --url http://localhost:8000 --rate 0.5 \
      --duration 60 --turns 3 --out results.json
"""

import argparse
import asyncio
import json
import random
import time

import httpx

_BASE_PROMPTS = [
    (
        "You are the on-call assistant for the ACME distributed systems team. "
        "Answer briefly and precisely. Prefer mechanisms over folklore, cite "
        "the subsystem involved, and when discussing performance always "
        "distinguish latency from throughput. If a question is ambiguous, "
        "state your assumption in one clause and answer anyway. Never pad "
        "answers with pleasantries. The team runs a paged-KV inference "
        "cluster, so memory arithmetic matters in every answer you give. "
    ),
    (
        "You are a study coach for an undergraduate operating systems course. "
        "Explain concepts with one concrete example each, in three sentences "
        "or fewer. Use analogies from everyday life sparingly and only when "
        "they map cleanly onto the mechanism. When a student asks something "
        "off-topic, redirect to the syllabus. The course covers scheduling, "
        "virtual memory, file systems, and concurrency, in that order, and "
        "exams reward mechanism-level understanding over memorized terms. "
    ),
    (
        "You are a code-review bot for a Go and Python monorepo. Point out "
        "correctness issues first, then performance, then style, as separate "
        "bullet points. Quote the smallest relevant snippet. Assume the "
        "reader knows the language; never explain syntax. Flag any shared "
        "mutable state, unbounded queue, or missing cancellation path — this "
        "codebase runs a streaming inference service where leaked requests "
        "hold GPU memory blocks until cancelled. "
    ),
]

# Long shared prefixes are what differentiate routing policies: ~900 tokens
# of system prompt means turn-1 prefill is expensive and reusable. Numbered
# "policy clauses" pad each base prompt to ~4000 chars without being pure
# repetition (repetition would be fine for the KV cache, but this keeps the
# workload honest-looking).
SYSTEM_PROMPTS = [
    base + "".join(
        f"Policy clause {i}: when scenario {i} arises, first identify the "
        f"resource under contention, then name the mechanism that arbitrates "
        f"it, and only then give the recommendation, numbered {i}. "
        for i in range(1, 25)
    )
    for base in _BASE_PROMPTS
]

QUESTIONS = [
    "What is a KV cache and why does it exist?",
    "Explain the difference between prefill and decode.",
    "Why is decode memory-bandwidth-bound?",
    "What problem does paging solve?",
    "What is copy-on-write?",
    "How does continuous batching differ from static batching?",
    "What is a radix tree used for in inference servers?",
    "Why do routers want cache locality?",
    "What is the convoy effect in scheduling?",
    "When does preemption beat queueing?",
    "What is chunked prefill?",
    "Explain grouped-query attention in one paragraph.",
]


async def run_conversation(
    client: httpx.AsyncClient, url: str, conv_id: int, turns: int,
    max_tokens: int, records: list, rng: random.Random,
):
    system = SYSTEM_PROMPTS[conv_id % len(SYSTEM_PROMPTS)]
    history = [{"role": "system", "content": system}]
    for turn in range(turns):
        history.append({"role": "user", "content": rng.choice(QUESTIONS)})
        rec = {
            "conv": conv_id, "turn": turn, "ok": False,
            "ttft_s": None, "itl_s": [], "latency_s": None,
            "worker": None, "chunks": 0,
        }
        t0 = time.perf_counter()
        text_parts = []
        try:
            async with client.stream(
                "POST", f"{url}/v1/chat/completions",
                json={
                    "conversation_id": f"conv-{conv_id}",
                    "messages": history,
                    "temperature": 0.7, "top_p": 0.8,
                    "max_tokens": max_tokens, "stream": True,
                    "seed": conv_id * 1000 + turn,
                },
                timeout=httpx.Timeout(10.0, read=300.0),
            ) as resp:
                rec["worker"] = resp.headers.get("x-routed-to")
                last = None
                async for line in resp.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    payload = json.loads(line[6:])
                    delta = payload["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    now = time.perf_counter()
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = now - t0
                    elif last is not None:
                        rec["itl_s"].append(now - last)
                    last = now
                    rec["chunks"] += 1
                    text_parts.append(delta)
            rec["latency_s"] = time.perf_counter() - t0
            rec["ok"] = rec["ttft_s"] is not None
        except Exception as e:
            rec["error"] = repr(e)
        records.append(rec)
        if not rec["ok"]:
            return
        history.append({"role": "assistant", "content": "".join(text_parts)})
        await asyncio.sleep(rng.uniform(0.2, 0.8))  # think time between turns


async def main_async(args):
    records: list = []
    rng = random.Random(args.seed)
    tasks = []
    async with httpx.AsyncClient() as client:
        t_end = time.perf_counter() + args.duration
        conv_id = 0
        while time.perf_counter() < t_end:
            tasks.append(asyncio.create_task(run_conversation(
                client, args.url, conv_id, args.turns,
                args.max_tokens, records, rng,
            )))
            conv_id += 1
            # Open loop: next arrival is exponential regardless of completions.
            await asyncio.sleep(rng.expovariate(args.rate))
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=600
        )
    return records


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def summarize(records, wall_s):
    ok = [r for r in records if r["ok"]]
    ttfts = [r["ttft_s"] for r in ok]
    itls = [x for r in ok for x in r["itl_s"]]
    return {
        "requests": len(records),
        "completed": len(ok),
        "throughput_rps": len(ok) / wall_s,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p99_s": percentile(ttfts, 99),
        "itl_p50_s": percentile(itls, 50),
        "itl_p99_s": percentile(itls, 99),
        "per_worker": {
            w: sum(1 for r in ok if r["worker"] == w)
            for w in sorted({r["worker"] for r in ok if r["worker"]})
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rate", type=float, default=0.5, help="conversation arrivals/sec")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    records = asyncio.run(main_async(args))
    wall = time.perf_counter() - t0
    summary = summarize(records, wall)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "records": records}, f)


if __name__ == "__main__":
    main()
