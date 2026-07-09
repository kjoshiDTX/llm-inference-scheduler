"""Week 11: the worker — an OpenAI-shaped serving layer around mini-engine.

- POST /v1/chat/completions (stream and non-stream)
- SSE streaming with incremental detokenization (engine side handles UTF-8)
- client disconnect -> engine.cancel -> blocks freed immediately
- GET /metrics (Prometheus text format), /health, /cache_state (for the router)

Run: WORKER_PORT=8001 .venv/bin/python -m uvicorn server.worker:app --port 8001
Config via env: WORKER_ID, NUM_BLOCKS, MAX_NUM_SEQS, MAX_BATCHED_TOKENS,
TORCH_THREADS, ENABLE_PREFIX_CACHE.
"""

import asyncio
import json
import os
import time
import uuid

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from mini_engine.engine import Engine
from mini_engine.sampling import SamplingParams
from mini_engine.scheduler import RequestState

WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")
MODEL_NAME = "mini-engine/qwen2.5-0.5b-instruct"

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

app = FastAPI(title="mini-engine worker")
engine: Engine | None = None


@app.on_event("startup")
def _startup():
    global engine
    engine = Engine(
        num_blocks=int(os.environ.get("NUM_BLOCKS", "1024")),
        max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
        max_batched_tokens=int(os.environ.get("MAX_BATCHED_TOKENS", "256")),
        enable_prefix_cache=os.environ.get("ENABLE_PREFIX_CACHE", "1") == "1",
    )
    engine.start()


@app.on_event("shutdown")
def _shutdown():
    if engine is not None:
        engine.stop()


def _params_from(body: dict) -> SamplingParams:
    return SamplingParams(
        temperature=float(body.get("temperature", 0.7)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0)),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        max_new_tokens=int(body.get("max_tokens", 256)),
        seed=body.get("seed"),
    )


def _chunk(chat_id: str, created: int, delta: dict, finish_reason=None) -> str:
    return "data: " + json.dumps({
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "worker_id": WORKER_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }) + "\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body["messages"]
    params = _params_from(body)
    stream = bool(body.get("stream", False))

    req = engine.submit_chat(messages, params)
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:20]}"
    created = int(time.time())

    if not stream:
        text_parts: list[str] = []
        finish = "stop"
        while True:
            ev = await asyncio.to_thread(req.events.get)
            if ev["type"] == "token":
                text_parts.append(ev["text"])
            else:
                finish = ev["reason"]
                break
        return JSONResponse({
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "worker_id": WORKER_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": "length" if finish == "length" else "stop",
            }],
            "usage": {
                "prompt_tokens": len(req.prompt_ids),
                "completion_tokens": len(req.output_ids),
                "total_tokens": req.total_len,
                "prefix_cached_tokens": req.num_prefix_cached,
            },
        })

    async def sse():
        try:
            yield _chunk(chat_id, created, {"role": "assistant", "content": ""})
            while True:
                # Poll with a timeout so we can notice client disconnects
                # even when no tokens are flowing.
                try:
                    ev = await asyncio.to_thread(req.events.get, True, 0.5)
                except Exception:  # queue.Empty
                    if await request.is_disconnected():
                        engine.cancel(req)
                        return
                    continue
                if ev["type"] == "token":
                    yield _chunk(chat_id, created, {"content": ev["text"]})
                else:
                    reason = "length" if ev["reason"] == "length" else "stop"
                    yield _chunk(chat_id, created, {}, finish_reason=reason)
                    yield "data: [DONE]\n\n"
                    return
        finally:
            # Generator torn down (client vanished mid-stream): free blocks now.
            if req.state not in (RequestState.FINISHED, RequestState.CANCELLED):
                engine.cancel(req)

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok", "worker_id": WORKER_ID, "queue_depth": engine.queue_depth()}


@app.get("/cache_state")
def cache_state():
    """Router-facing summary of what this worker has cached."""
    m = engine.metrics()
    return {
        "worker_id": WORKER_ID,
        "queue_depth": m["num_waiting"] + m["num_running"],
        "free_blocks": m["num_free_blocks"],
        "total_blocks": m["num_total_blocks"],
        "radix_tree_blocks": m.get("radix_tree_blocks", 0),
    }


@app.get("/metrics")
def metrics():
    m = engine.metrics()
    lines = []
    for key, val in m.items():
        name = f"mini_engine_{key}"
        lines.append(f"# TYPE {name} gauge")
        lines.append(f'{name}{{worker_id="{WORKER_ID}"}} {val}')
    return PlainTextResponse("\n".join(lines) + "\n")
