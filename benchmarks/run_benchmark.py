"""Week 14: the headline experiment.

Three routing policies — round_robin, hash affinity (the old design),
cache_aware (overlap - queue depth) — under the same open-loop multi-turn
workload with shared system prompts, at increasing offered load. Workers are
restarted between policies so every run starts with cold caches.

Produces benchmarks/results/results.json and headline.png.

Usage: .venv/bin/python benchmarks/run_benchmark.py [--rates 0.15,0.3,0.5]
       [--duration 45] [--quick]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
RESULTS_DIR = ROOT / "benchmarks" / "results"

WORKER_PORTS = [8001, 8002]
ROUTER_PORT = 8000
POLICIES = ["round_robin", "hash", "cache_aware"]


def wait_healthy(url: str, timeout: float = 120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{url} not healthy after {timeout}s")


def start_cluster(policy: str):
    procs = []
    for i, port in enumerate(WORKER_PORTS):
        env = dict(
            os.environ,
            WORKER_ID=f"w{i+1}",
            NUM_BLOCKS="1024",
            TORCH_THREADS="3",
            MAX_NUM_SEQS="8",
            MAX_BATCHED_TOKENS="256",
        )
        procs.append(subprocess.Popen(
            [PY, "-m", "uvicorn", "server.worker:app",
             "--port", str(port), "--log-level", "warning"],
            cwd=ROOT, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    workers = ",".join(f"http://localhost:{p}" for p in WORKER_PORTS)
    procs.append(subprocess.Popen(
        [str(ROOT / "router" / "router"),
         "-listen", f":{ROUTER_PORT}", "-policy", policy,
         "-alpha", "1", "-beta", "1000", "-workers", workers],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ))
    for port in WORKER_PORTS:
        wait_healthy(f"http://localhost:{port}")
    wait_healthy(f"http://localhost:{ROUTER_PORT}")
    return procs


def stop_cluster(procs):
    for p in procs:
        p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    time.sleep(2)


def run_load(rate: float, duration: float, out: Path):
    subprocess.run(
        [PY, str(ROOT / "loadgen" / "loadgen.py"),
         "--url", f"http://localhost:{ROUTER_PORT}",
         "--rate", str(rate), "--duration", str(duration),
         "--turns", "3", "--max-tokens", "24", "--out", str(out)],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    with open(out) as f:
        return json.load(f)["summary"]


def plot(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"round_robin": "#888888", "hash": "#e0801f", "cache_aware": "#1f77b4"}
    labels = {
        "round_robin": "round-robin",
        "hash": "hash affinity (old router)",
        "cache_aware": "cache-aware (overlap − queue)",
    }
    for policy, points in results.items():
        rates = [p["rate"] for p in points]
        axes[0].plot(rates, [p["summary"]["ttft_p99_s"] for p in points],
                     "o-", color=colors[policy], label=labels[policy])
        axes[0].plot(rates, [p["summary"]["ttft_p50_s"] for p in points],
                     "o--", color=colors[policy], alpha=0.4)
        axes[1].plot(rates, [p["summary"]["throughput_rps"] for p in points],
                     "o-", color=colors[policy], label=labels[policy])
    axes[0].set_xlabel("offered load (conversations/s)")
    axes[0].set_ylabel("TTFT p99 (solid) / p50 (dashed), seconds")
    axes[0].set_title("TTFT vs offered load — 2 mini-engine workers")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("offered load (conversations/s)")
    axes[1].set_ylabel("completed requests/s")
    axes[1].set_title("Goodput vs offered load")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Router policy comparison — multi-turn + shared system prompts, "
                 "open-loop Poisson arrivals", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"chart -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.15,0.3,0.5")
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--quick", action="store_true",
                    help="one rate, short duration, for smoke testing")
    args = ap.parse_args()

    rates = [float(r) for r in args.rates.split(",")]
    duration = args.duration
    if args.quick:
        rates, duration = [0.3], 20

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, list] = {}
    for policy in POLICIES:
        results[policy] = []
        for rate in rates:
            print(f"=== {policy} @ {rate} conv/s ===", flush=True)
            procs = start_cluster(policy)
            try:
                out = RESULTS_DIR / f"{policy}_{rate}.json"
                summary = run_load(rate, duration, out)
                results[policy].append({"rate": rate, "summary": summary})
                print(json.dumps(summary, indent=2), flush=True)
            finally:
                stop_cluster(procs)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    plot(results, RESULTS_DIR / "headline.png")


if __name__ == "__main__":
    main()
