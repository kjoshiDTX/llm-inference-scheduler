"""mini-engine v0.1: chat with Qwen2.5-0.5B end-to-end through our own code.

Usage: python scripts/chat.py [--prompt "..."] [--greedy]
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch

from mini_engine.generate import generate
from mini_engine.model.loader import load_weights
from mini_engine.model.qwen2 import Qwen2Model
from mini_engine.sampling import SamplingParams
from mini_engine.tokenizer.qwen_tokenizer import QwenTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    print("loading weights...", file=sys.stderr)
    config, weights = load_weights(device="cpu", dtype=torch.float32)
    model = Qwen2Model(config, weights)
    tok = QwenTokenizer.from_model()
    params = SamplingParams(
        temperature=0.0 if args.greedy else 0.7,
        top_p=0.8,
        max_new_tokens=args.max_tokens,
    )

    history: list[dict] = []

    def run_turn(user_text: str):
        history.append({"role": "user", "content": user_text})
        ids = tok.apply_chat_template(history)
        print("assistant: ", end="", flush=True)
        result = generate(
            model, tok, ids, params, on_token=lambda s: print(s, end="", flush=True)
        )
        print()
        print(
            f"  [ttft {result.ttft_s:.2f}s, {result.decode_tps:.1f} tok/s decode]",
            file=sys.stderr,
        )
        history.append({"role": "assistant", "content": result.text})

    if args.prompt:
        run_turn(args.prompt)
        return

    print("mini-engine v0.1 — ctrl-d to exit", file=sys.stderr)
    while True:
        try:
            user = input("you: ")
        except EOFError:
            break
        if user.strip():
            run_turn(user)


if __name__ == "__main__":
    main()
