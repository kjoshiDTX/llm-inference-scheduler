"""Week 8: continuous batching (Orca) + chunked prefill (Sarathi-Serve).

Week 5's static batching has the convoy problem: the batch finishes when its
longest member does. The fix is scheduling at *iteration* granularity — every
step, the batch is rebuilt: ongoing decodes get one token each, waiting
requests are admitted if blocks allow, and long prefills are chunked so they
never stall other requests' inter-token latency.

Preemption: when memory runs out mid-generation we evict radix-tree blocks
first, then preempt the most-recently-admitted running request (free its
blocks, recompute later — the recompute-vs-swap tradeoff from the Orca/vLLM
papers; recompute wins at small scale and needs no host-memory plumbing).
"""

import enum
import queue
import time
from dataclasses import dataclass, field

import torch

from .model.qwen2 import Qwen2Model, SequenceForwardInfo
from .paged.block_manager import BlockManager, OutOfBlocks, PagedKVCache
from .paged.radix_cache import RadixCache
from .sampling import SamplingParams, sample
from .tokenizer.qwen_tokenizer import IncrementalDetokenizer, QwenTokenizer


class RequestState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    params: SamplingParams
    state: RequestState = RequestState.WAITING
    output_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    num_prefix_cached: int = 0     # tokens skipped via radix match (stats)
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    detok: IncrementalDetokenizer | None = None
    generator: torch.Generator | None = None
    arrived_at: float = field(default_factory=time.monotonic)
    admitted_at: float | None = None
    first_token_at: float | None = None
    finish_reason: str | None = None
    preemptions: int = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)



class Scheduler:
    def __init__(
        self,
        model: Qwen2Model,
        tokenizer: QwenTokenizer,
        block_manager: BlockManager,
        max_num_seqs: int = 8,
        max_batched_tokens: int = 256,
        enable_prefix_cache: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.mgr = block_manager
        self.kv = PagedKVCache(block_manager)
        self.radix = RadixCache(block_manager) if enable_prefix_cache else None
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens

        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.metrics = {
            "requests_finished": 0,
            "requests_cancelled": 0,
            "preemptions": 0,
            "steps": 0,
            "tokens_generated": 0,
            "prefill_tokens_computed": 0,
            "prefix_cache_hit_tokens": 0,
        }

    # --- admission / memory -------------------------------------------------

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.mgr.block_size - 1) // self.mgr.block_size

    def _try_allocate(self, n: int) -> list[int] | None:
        """Allocate n blocks, evicting from the radix tree if needed."""
        if self.mgr.num_free < n and self.radix is not None:
            self.radix.evict(n - self.mgr.num_free)
        if self.mgr.num_free < n:
            return None
        return [self.mgr.allocate() for _ in range(n)]

    def _admit(self, req: Request):
        feed = req.all_ids  # includes generated tokens when resuming preempted
        matched_blocks: list[int] = []
        matched = 0
        if self.radix is not None:
            blocks, matched = self.radix.match_prefix(feed)
            # Always leave >=1 token to actually compute (we need its logits).
            while matched >= len(feed):
                blocks = blocks[:-1]
                matched -= self.mgr.block_size
            matched_blocks = blocks

        new_blocks = self._try_allocate(self._blocks_for(len(feed) + 1) - len(matched_blocks))
        if new_blocks is None:
            return False
        if self.radix is not None:
            self.radix.adopt(matched_blocks)
        req.block_table = matched_blocks + new_blocks
        req.num_computed_tokens = matched
        req.num_prefix_cached += matched
        self.metrics["prefix_cache_hit_tokens"] += matched
        self.kv.add_sequence(req.req_id, req.block_table, matched)
        req.state = RequestState.RUNNING
        if req.admitted_at is None:
            req.admitted_at = time.monotonic()
        self.running.append(req)
        return True

    def _release_blocks(self, req: Request):
        for b in req.block_table:
            self.mgr.decref(b)
        req.block_table = []
        self.kv.free_sequence(req.req_id)

    def _preempt_one(self) -> bool:
        """Preempt the most recently admitted running request (recompute)."""
        candidates = [r for r in self.running if r.state is RequestState.RUNNING]
        if len(candidates) <= 1:
            return False
        victim = candidates[-1]
        self._release_blocks(victim)
        victim.num_computed_tokens = 0
        victim.state = RequestState.WAITING
        victim.preemptions += 1
        self.metrics["preemptions"] += 1
        self.running.remove(victim)
        self.waiting.insert(0, victim)
        return True

    def _finish(self, req: Request, reason: str):
        if self.radix is not None and req.state is not RequestState.CANCELLED:
            # Donate this sequence's full blocks to the prefix cache. The tree
            # increfs what it references; our decref below keeps net ownership
            # correct.
            self.radix.insert(req.all_ids[: req.num_computed_tokens], req.block_table)
        self._release_blocks(req)
        req.state = (
            RequestState.CANCELLED if reason == "cancelled" else RequestState.FINISHED
        )
        req.finish_reason = reason
        key = "requests_cancelled" if reason == "cancelled" else "requests_finished"
        self.metrics[key] += 1
        if req.detok is not None and reason != "cancelled":
            tail = req.detok.flush()
            if tail:
                req.events.put({"type": "token", "text": tail})
        req.events.put({"type": "finish", "reason": reason})

    def cancel(self, req: Request):
        if req.state in (RequestState.FINISHED, RequestState.CANCELLED):
            return
        if req in self.waiting:
            self.waiting.remove(req)
        if req in self.running:
            self.running.remove(req)
        req.state = RequestState.CANCELLED
        self._finish(req, "cancelled")

    # --- the iteration loop ---------------------------------------------------

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    @torch.inference_mode()
    def step(self):
        """One iteration: admit, ensure memory, forward one flat batch, sample."""
        self.metrics["steps"] += 1

        # 1. Admit while there is room (iteration-level admission — the whole
        #    point of continuous batching).
        while self.waiting and len(self.running) < self.max_num_seqs:
            if not self._admit(self.waiting[0]):
                break
            self.waiting.pop(0)

        if not self.running:
            return

        # 2. Ensure every scheduled sequence has a block for its next token;
        #    preempt (recompute) if allocation fails even after radix eviction.
        stalled: set[int] = set()
        for req in list(self.running):
            needed = self._blocks_for(req.total_len + 1) - len(req.block_table)
            while needed > 0 and req.state is RequestState.RUNNING:
                got = self._try_allocate(needed)
                if got is not None:
                    req.block_table.extend(got)
                    needed = 0
                    break
                if not self._preempt_one():
                    # Can't free memory (we're the only candidate): sit this
                    # step out and retry when other sequences release blocks.
                    stalled.add(req.req_id)
                    break
                needed = self._blocks_for(req.total_len + 1) - len(req.block_table)

        active = [
            r for r in self.running
            if r.state is RequestState.RUNNING and r.req_id not in stalled
        ]
        if not active:
            return

        # 3. Build the flat batch. Every request feeds its known-but-uncomputed
        #    tokens: 1 for an ongoing decode (the token sampled last step),
        #    a chunk for prefill/recompute. Ongoing decodes are budgeted first
        #    so a long incoming prefill can never stall their ITL — that
        #    ordering IS chunked prefill's TTFT-vs-ITL knob.
        budget = self.max_batched_tokens
        flat_ids: list[int] = []
        infos: list[SequenceForwardInfo] = []
        scheduled: list[Request] = []
        ordered = sorted(active, key=lambda r: 0 if r.output_ids else 1)

        for req in ordered:
            if budget <= 0:
                break
            remaining = req.total_len - req.num_computed_tokens
            chunk = min(remaining, budget)
            budget -= chunk
            self.kv.ensure_writable_tail(req.req_id)
            start = req.num_computed_tokens
            feed = req.all_ids[start : start + chunk]
            infos.append(
                SequenceForwardInfo(
                    seq_id=req.req_id,
                    flat_start=len(flat_ids),
                    num_tokens=chunk,
                    positions=torch.arange(start, start + chunk),
                )
            )
            flat_ids.extend(feed)
            scheduled.append(req)

        input_ids = torch.tensor(flat_ids, device=self.model.device)
        logits = self.model.forward(input_ids, infos, self.kv)

        # 4. Sample for sequences that reached the end of their known tokens.
        for i, req in enumerate(scheduled):
            n_new = infos[i].num_tokens
            if not req.output_ids:
                self.metrics["prefill_tokens_computed"] += n_new
            req.num_computed_tokens += n_new
            if req.num_computed_tokens < req.total_len:
                continue  # chunked prefill / recompute not finished; no sampling
            token = sample(logits[i], req.params, req.output_ids, req.generator)
            req.output_ids.append(token)
            self.metrics["tokens_generated"] += 1
            if req.first_token_at is None:
                req.first_token_at = time.monotonic()

            stop_ids = req.params.stop_token_ids or self.tokenizer.eos_ids
            if token in stop_ids:
                self._finish(req, "stop")
                continue
            if req.detok is not None:
                piece = req.detok.push(token)
                if piece:
                    req.events.put({"type": "token", "text": piece})
            if len(req.output_ids) >= req.params.max_new_tokens:
                self._finish(req, "length")

        self.running = [r for r in self.running if r.state is RequestState.RUNNING]
