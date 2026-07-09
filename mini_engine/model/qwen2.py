"""Qwen2 forward pass from raw tensors. No transformers modeling code.

Architecture (Qwen2.5-0.5B): embedding -> 24 x [RMSNorm -> GQA attention
(14Q/2KV, RoPE, qkv bias) -> RMSNorm -> SwiGLU MLP] -> RMSNorm -> LM head
(tied to embeddings).

The forward pass operates on a flat token batch (vLLM-style): sequences of
different lengths are concatenated, and attention runs per-sequence against
K/V fetched from a cache object. That one design choice lets the same code
serve single-sequence prefill, batched decode, chunked prefill, and the
paged cache without special cases.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import Qwen2Config


@dataclass
class SequenceForwardInfo:
    """One sequence's slice of the flat token batch for a forward call."""

    seq_id: int
    flat_start: int       # start offset in the flat input_ids batch
    num_tokens: int       # number of NEW tokens for this sequence this step
    positions: torch.Tensor  # [num_tokens] absolute positions of the new tokens


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding:
    """RoPE: rotate (dim i, dim i+d/2) pairs by position-dependent angles so
    that q.k dot products depend only on relative position."""

    def __init__(self, head_dim: int, theta: float, device, dtype):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        self.inv_freq = inv_freq
        self.device = device
        self.dtype = dtype
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len = 0

    def _ensure(self, max_pos: int):
        if max_pos <= self._cached_len:
            return
        n = max(max_pos, 4096, self._cached_len * 2)
        t = torch.arange(n, device=self.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [n, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)         # [n, head_dim]
        self._cos = emb.cos().to(self.dtype)
        self._sin = emb.sin().to(self.dtype)
        self._cached_len = n

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """q: [n_tokens, n_q_heads, head_dim], k: [n_tokens, n_kv_heads, head_dim]."""
        self._ensure(int(positions.max().item()) + 1)
        cos = self._cos[positions].unsqueeze(1)  # [n_tokens, 1, head_dim]
        sin = self._sin[positions].unsqueeze(1)
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot


class Qwen2Model:
    def __init__(self, config: Qwen2Config, weights: dict[str, torch.Tensor]):
        self.config = config
        self.w = weights
        first = next(iter(weights.values()))
        self.device = first.device
        self.dtype = first.dtype
        self.rope = RotaryEmbedding(
            config.head_dim, config.rope_theta, self.device, self.dtype
        )
        self.scale = config.head_dim ** -0.5
        # Tied embeddings: LM head reuses the embedding matrix.
        self.lm_head = (
            weights["model.embed_tokens.weight"]
            if config.tie_word_embeddings
            else weights["lm_head.weight"]
        )

    def _attention(
        self,
        q: torch.Tensor,  # [num_new, n_q_heads, head_dim]
        k_all: torch.Tensor,  # [total_len, n_kv_heads, head_dim] (cache incl. new)
        v_all: torch.Tensor,
        num_prior: int,  # tokens already in cache before this step's new ones
    ) -> torch.Tensor:
        cfg = self.config
        group = cfg.num_attention_heads // cfg.num_key_value_heads  # 7 for Qwen2.5-0.5B

        # GQA: each KV head serves `group` query heads.
        k = k_all.repeat_interleave(group, dim=1)  # [total_len, n_q_heads, head_dim]
        v = v_all.repeat_interleave(group, dim=1)

        q = q.transpose(0, 1)      # [heads, num_new, head_dim]
        k = k.permute(1, 2, 0)     # [heads, head_dim, total_len]
        v = v.transpose(0, 1)      # [heads, total_len, head_dim]

        scores = torch.bmm(q, k) * self.scale  # [heads, num_new, total_len]

        # Causal mask: new token at absolute index (num_prior + i) may attend
        # to cache positions 0..num_prior+i.
        num_new, total_len = q.shape[1], k.shape[2]
        if num_new > 1:
            q_idx = torch.arange(num_new, device=q.device).unsqueeze(1) + num_prior
            kv_idx = torch.arange(total_len, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(kv_idx > q_idx, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.bmm(attn, v)  # [heads, num_new, head_dim]
        return out.transpose(0, 1).reshape(num_new, -1)  # [num_new, hidden]

    def forward(
        self,
        input_ids: torch.Tensor,  # [total_tokens] flat across sequences
        seq_infos: list[SequenceForwardInfo],
        kv_cache,  # object with append_and_fetch(layer, seq_id, k, v) -> (K, V, num_prior)
    ) -> torch.Tensor:
        """Returns logits for the LAST new token of each sequence: [len(seq_infos), vocab]."""
        cfg = self.config
        w = self.w

        positions = torch.cat([s.positions for s in seq_infos]).to(self.device)
        h = w["model.embed_tokens.weight"][input_ids]  # [total_tokens, hidden]

        for layer in range(cfg.num_hidden_layers):
            p = f"model.layers.{layer}"
            residual = h
            x = rms_norm(h, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)

            q = F.linear(x, w[f"{p}.self_attn.q_proj.weight"], w[f"{p}.self_attn.q_proj.bias"])
            k = F.linear(x, w[f"{p}.self_attn.k_proj.weight"], w[f"{p}.self_attn.k_proj.bias"])
            v = F.linear(x, w[f"{p}.self_attn.v_proj.weight"], w[f"{p}.self_attn.v_proj.bias"])

            n = x.shape[0]
            q = q.view(n, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(n, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(n, cfg.num_key_value_heads, cfg.head_dim)
            q, k = self.rope.apply(q, k, positions)

            # Per-sequence attention against the cache.
            outs = []
            for s in seq_infos:
                sl = slice(s.flat_start, s.flat_start + s.num_tokens)
                k_all, v_all, num_prior = kv_cache.append_and_fetch(
                    layer, s.seq_id, k[sl], v[sl]
                )
                outs.append(self._attention(q[sl], k_all, v_all, num_prior))
            attn_out = torch.cat(outs, dim=0)

            h = residual + F.linear(attn_out, w[f"{p}.self_attn.o_proj.weight"])

            residual = h
            x = rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            gate = F.linear(x, w[f"{p}.mlp.gate_proj.weight"])
            up = F.linear(x, w[f"{p}.mlp.up_proj.weight"])
            h = residual + F.linear(F.silu(gate) * up, w[f"{p}.mlp.down_proj.weight"])

        h = rms_norm(h, w["model.norm.weight"], cfg.rms_norm_eps)

        # Only the last new token of each sequence produces a next-token logit
        # we care about (saves the [total_tokens, 151936] matmul during prefill).
        last_idx = torch.tensor(
            [s.flat_start + s.num_tokens - 1 for s in seq_infos], device=self.device
        )
        return F.linear(h[last_idx], self.lm_head)  # [num_seqs, vocab]

    def forward_all_logits(
        self, input_ids: torch.Tensor, seq_infos: list[SequenceForwardInfo], kv_cache
    ) -> torch.Tensor:
        """Full-sequence logits, used only by the HF-parity test."""
        cfg, w = self.config, self.w
        positions = torch.cat([s.positions for s in seq_infos]).to(self.device)
        h = w["model.embed_tokens.weight"][input_ids]
        for layer in range(cfg.num_hidden_layers):
            p = f"model.layers.{layer}"
            residual = h
            x = rms_norm(h, w[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)
            q = F.linear(x, w[f"{p}.self_attn.q_proj.weight"], w[f"{p}.self_attn.q_proj.bias"])
            k = F.linear(x, w[f"{p}.self_attn.k_proj.weight"], w[f"{p}.self_attn.k_proj.bias"])
            v = F.linear(x, w[f"{p}.self_attn.v_proj.weight"], w[f"{p}.self_attn.v_proj.bias"])
            n = x.shape[0]
            q = q.view(n, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(n, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(n, cfg.num_key_value_heads, cfg.head_dim)
            q, k = self.rope.apply(q, k, positions)
            outs = []
            for s in seq_infos:
                sl = slice(s.flat_start, s.flat_start + s.num_tokens)
                k_all, v_all, num_prior = kv_cache.append_and_fetch(
                    layer, s.seq_id, k[sl], v[sl]
                )
                outs.append(self._attention(q[sl], k_all, v_all, num_prior))
            h = residual + F.linear(torch.cat(outs, dim=0), w[f"{p}.self_attn.o_proj.weight"])
            residual = h
            x = rms_norm(h, w[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            gate = F.linear(x, w[f"{p}.mlp.gate_proj.weight"])
            up = F.linear(x, w[f"{p}.mlp.up_proj.weight"])
            h = residual + F.linear(F.silu(gate) * up, w[f"{p}.mlp.down_proj.weight"])
        h = rms_norm(h, w["model.norm.weight"], cfg.rms_norm_eps)
        return F.linear(h, self.lm_head)
