"""The engine facade: model + paged memory + scheduler behind a thread-safe
submit/cancel API. The server (server/worker.py) talks only to this class.

Threading model: one engine thread runs the step loop; request submission and
token consumption happen from other threads (FastAPI handlers) through
queue.Queue. This mirrors vLLM's single-threaded core loop + async frontend.
"""

import itertools
import threading

import torch

from .kv_cache import ContiguousKVCache  # noqa: F401  (re-export for tests)
from .model.config import Qwen2Config
from .model.loader import MODEL_ID, load_weights
from .model.qwen2 import Qwen2Model
from .paged.block_manager import BlockManager
from .sampling import SamplingParams
from .scheduler import Request, RequestState, Scheduler
from .tokenizer.qwen_tokenizer import IncrementalDetokenizer, QwenTokenizer


class Engine:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        dtype=torch.float32,
        num_blocks: int = 1024,
        block_size: int = 16,
        max_num_seqs: int = 8,
        max_batched_tokens: int = 256,
        enable_prefix_cache: bool = True,
    ):
        config, weights = load_weights(model_id, device=device, dtype=dtype)
        self.config: Qwen2Config = config
        self.model = Qwen2Model(config, weights)
        self.tokenizer = QwenTokenizer.from_model(model_id)
        self.block_manager = BlockManager(
            num_blocks, block_size,
            config.num_hidden_layers, config.num_key_value_heads, config.head_dim,
            device=device, dtype=dtype,
        )
        self.scheduler = Scheduler(
            self.model, self.tokenizer, self.block_manager,
            max_num_seqs=max_num_seqs,
            max_batched_tokens=max_batched_tokens,
            enable_prefix_cache=enable_prefix_cache,
        )
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._requests: dict[int, Request] = {}
        self._stop = False
        self._thread: threading.Thread | None = None

    # --- public API (any thread) ---------------------------------------------

    def submit_chat(self, messages: list[dict], params: SamplingParams) -> Request:
        prompt_ids = self.tokenizer.apply_chat_template(messages)
        return self.submit(prompt_ids, params)

    def submit(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        req = Request(req_id=next(self._ids), prompt_ids=prompt_ids, params=params)
        req.detok = IncrementalDetokenizer(self.tokenizer)
        if params.seed is not None:
            req.generator = torch.Generator().manual_seed(params.seed)
        with self._work:
            self.scheduler.waiting.append(req)
            self._requests[req.req_id] = req
            self._work.notify()
        return req

    def cancel(self, req: Request):
        with self._work:
            self.scheduler.cancel(req)

    def metrics(self) -> dict:
        with self._lock:
            m = dict(self.scheduler.metrics)
            m["num_waiting"] = len(self.scheduler.waiting)
            m["num_running"] = len(self.scheduler.running)
            m["num_free_blocks"] = self.block_manager.num_free
            m["num_total_blocks"] = self.block_manager.num_blocks
            if self.scheduler.radix is not None:
                m["radix_tree_blocks"] = self.scheduler.radix.num_tree_blocks
                m["radix_query_tokens"] = self.scheduler.radix.query_tokens
                m["radix_hit_tokens"] = self.scheduler.radix.hit_tokens
            return m

    def queue_depth(self) -> int:
        with self._lock:
            return len(self.scheduler.waiting) + len(self.scheduler.running)

    # --- engine loop -----------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        with self._work:
            self._stop = True
            self._work.notify()
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self):
        while True:
            with self._work:
                while not self._stop and not self.scheduler.has_work:
                    self._work.wait()
                if self._stop:
                    return
                self.scheduler.step()
                # Drop finished requests from the registry.
                done = [
                    rid for rid, r in self._requests.items()
                    if r.state in (RequestState.FINISHED, RequestState.CANCELLED)
                ]
                for rid in done:
                    del self._requests[rid]

    # --- synchronous convenience (tests, offline batch) -------------------------

    def generate_sync(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        req = self.submit(prompt_ids, params)
        while req.state not in (RequestState.FINISHED, RequestState.CANCELLED):
            with self._lock:
                self.scheduler.step()
        return req
