"""Week 3: the contiguous per-sequence KV cache.

Prefill = fill the cache for the whole prompt in one batched forward.
Decode = append one (K, V) row per layer per step and attend over everything.

Why it exists: without a cache, generating token n+1 re-runs attention over
all n prior tokens *and* recomputes their K/V projections — O(n^2) work per
token. With the cache, each new token costs one row of K/V per layer.

The cost: memory. kv_bytes_per_token for Qwen2.5-0.5B fp16 is 12 KB/token
(2 x 24 layers x 2 kv_heads x 64 head_dim x 2 bytes). Contiguous allocation
also fragments and can't share prefixes — that's what paged/ fixes in Week 6.
"""

import torch


class ContiguousKVCache:
    """Per-sequence, dynamically grown K/V tensors. Simple and correct;
    fragmentation and zero sharing are the point (paged cache fixes both)."""

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 device="cpu", dtype=torch.float32, initial_capacity: int = 256):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.initial_capacity = initial_capacity
        # seq_id -> per-layer (k_buf, v_buf); plus current length per seq
        self._k: dict[int, list[torch.Tensor]] = {}
        self._v: dict[int, list[torch.Tensor]] = {}
        self._len: dict[int, int] = {}

    def add_sequence(self, seq_id: int):
        cap = self.initial_capacity
        shape = (cap, self.num_kv_heads, self.head_dim)
        self._k[seq_id] = [torch.empty(shape, device=self.device, dtype=self.dtype)
                           for _ in range(self.num_layers)]
        self._v[seq_id] = [torch.empty(shape, device=self.device, dtype=self.dtype)
                           for _ in range(self.num_layers)]
        self._len[seq_id] = 0

    def free_sequence(self, seq_id: int):
        self._k.pop(seq_id, None)
        self._v.pop(seq_id, None)
        self._len.pop(seq_id, None)

    def seq_len(self, seq_id: int) -> int:
        return self._len[seq_id]

    def _grow(self, seq_id: int, needed: int):
        cur_cap = self._k[seq_id][0].shape[0]
        if needed <= cur_cap:
            return
        new_cap = max(needed, cur_cap * 2)
        for layer in range(self.num_layers):
            for bufs in (self._k, self._v):
                old = bufs[seq_id][layer]
                new = torch.empty((new_cap, *old.shape[1:]), device=old.device, dtype=old.dtype)
                new[: self._len[seq_id]] = old[: self._len[seq_id]]
                bufs[seq_id][layer] = new

    def append_and_fetch(self, layer: int, seq_id: int,
                         k_new: torch.Tensor, v_new: torch.Tensor):
        """Append this step's K/V rows for `layer`, return (K_total, V_total, num_prior).

        num_prior is the sequence length *before* this append — the model uses
        it to build the causal mask for multi-token (prefill/chunked) steps.
        `_len` is advanced only on the last layer so every layer sees the same
        num_prior within a single forward pass.
        """
        n_new = k_new.shape[0]
        num_prior = self._len[seq_id]
        self._grow(seq_id, num_prior + n_new)
        self._k[seq_id][layer][num_prior : num_prior + n_new] = k_new
        self._v[seq_id][layer][num_prior : num_prior + n_new] = v_new
        if layer == self.num_layers - 1:
            self._len[seq_id] = num_prior + n_new
        total = num_prior + n_new
        return (self._k[seq_id][layer][:total],
                self._v[seq_id][layer][:total],
                num_prior)


class NoCache:
    """Week 3's 'naive fail' baseline: recompute everything, every step.

    Exists so the with/without-cache benchmark is one flag. append_and_fetch
    returns only what was passed in, so the caller must re-feed the entire
    sequence each step.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers

    def add_sequence(self, seq_id: int):
        pass

    def free_sequence(self, seq_id: int):
        pass

    def seq_len(self, seq_id: int) -> int:
        return 0

    def append_and_fetch(self, layer, seq_id, k_new, v_new):
        return k_new, v_new, 0
