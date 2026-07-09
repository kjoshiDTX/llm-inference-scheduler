"""Week 4: logits -> token, with controllable randomness.

Order of operations mirrors production engines (vLLM's sampler):
repetition penalty -> temperature -> top-k -> top-p -> sample.
In vLLM this all runs on-GPU because shipping [vocab]-sized logits to CPU
every step would dominate inter-token latency; here the whole model runs
on one device so it's moot, but the structure is the same.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    repetition_penalty: float = 1.0
    max_new_tokens: int = 256
    stop_token_ids: set = field(default_factory=set)
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def sample(
    logits: torch.Tensor,  # [vocab]
    params: SamplingParams,
    generated_ids: list[int],
    generator: torch.Generator | None = None,
) -> int:
    logits = logits.float()

    if params.repetition_penalty != 1.0 and generated_ids:
        prev = torch.tensor(sorted(set(generated_ids)), device=logits.device)
        scores = logits[prev]
        # Penalize: positive logits divided, negative multiplied (CTRL-style).
        scores = torch.where(
            scores > 0, scores / params.repetition_penalty, scores * params.repetition_penalty
        )
        logits[prev] = scores

    if params.greedy:
        return int(torch.argmax(logits).item())

    logits = logits / params.temperature

    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.shape[-1])).values[-1]
        logits[logits < kth] = float("-inf")

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Keep the smallest set whose cumulative prob exceeds top_p
        # (always keep the first token).
        cutoff = cumulative - probs >= params.top_p
        sorted_logits[cutoff] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(
            0, sorted_idx, sorted_logits
        )

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())
