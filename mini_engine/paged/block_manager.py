"""Week 6: PagedAttention's memory system, simplified.

The founding observation of vLLM: contiguous per-request KV allocation is
the same problem virtual memory solved in the 1960s. Fixed-size blocks
(here 16 tokens) kill external fragmentation, cap internal fragmentation at
block_size-1 tokens per sequence, and — the part that matters for routing —
make the cache *shareable*: two sequences with the same prefix can point at
the same physical blocks, with ref counts and copy-on-write.

Physical layout: one K and one V pool per layer, shape
[num_blocks, block_size, kv_heads, head_dim]. All layers share one block
table per sequence (same as vLLM): logical block i of a sequence lives at
the same physical block index in every layer's pool.
"""

import torch


class OutOfBlocks(Exception):
    """Allocation failed; scheduler must preempt or wait."""


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device="cpu",
        dtype=torch.float32,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v_pool = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.ref_count = [0] * num_blocks
        self.free_list = list(range(num_blocks - 1, -1, -1))  # pop() from the end

    @property
    def num_free(self) -> int:
        return len(self.free_list)

    def allocate(self) -> int:
        if not self.free_list:
            raise OutOfBlocks()
        block = self.free_list.pop()
        assert self.ref_count[block] == 0
        self.ref_count[block] = 1
        return block

    def incref(self, block: int):
        assert self.ref_count[block] > 0, f"incref on free block {block}"
        self.ref_count[block] += 1

    def decref(self, block: int):
        assert self.ref_count[block] > 0, f"decref on free block {block}"
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.free_list.append(block)

    def copy_block(self, src: int) -> int:
        """Copy-on-write: give the caller a private copy of a shared block."""
        dst = self.allocate()
        for layer in range(self.num_layers):
            self.k_pool[layer][dst] = self.k_pool[layer][src]
            self.v_pool[layer][dst] = self.v_pool[layer][src]
        return dst


class PagedKVCache:
    """The model-facing adapter: same append_and_fetch contract as
    ContiguousKVCache, but reads/writes go through block tables.

    The scheduler owns block *allocation* (it must know about OutOfBlocks to
    preempt); this adapter only writes into already-allocated blocks.
    """

    def __init__(self, manager: BlockManager):
        self.mgr = manager
        self.block_tables: dict[int, list[int]] = {}
        self._len: dict[int, int] = {}

    def add_sequence(self, seq_id: int, block_table: list[int], num_cached_tokens: int):
        """num_cached_tokens = tokens already present via prefix-cache reuse."""
        self.block_tables[seq_id] = block_table
        self._len[seq_id] = num_cached_tokens

    def free_sequence(self, seq_id: int):
        self.block_tables.pop(seq_id, None)
        self._len.pop(seq_id, None)

    def seq_len(self, seq_id: int) -> int:
        return self._len[seq_id]

    def ensure_writable_tail(self, seq_id: int):
        """COW check before a step: if the block we are about to append into
        is shared (ref > 1), replace it with a private copy."""
        length = self._len[seq_id]
        table = self.block_tables[seq_id]
        idx = length // self.mgr.block_size
        if idx < len(table) and length % self.mgr.block_size != 0:
            block = table[idx]
            if self.mgr.ref_count[block] > 1:
                table[idx] = self.mgr.copy_block(block)
                self.mgr.decref(block)

    def append_and_fetch(self, layer: int, seq_id: int, k_new, v_new):
        bs = self.mgr.block_size
        table = self.block_tables[seq_id]
        num_prior = self._len[seq_id]
        n_new = k_new.shape[0]
        total = num_prior + n_new
        assert total <= len(table) * bs, (
            f"seq {seq_id}: {total} tokens but only {len(table)} blocks allocated"
        )

        # Scatter new rows into blocks.
        pos = num_prior
        written = 0
        while written < n_new:
            block = table[pos // bs]
            off = pos % bs
            n = min(bs - off, n_new - written)
            self.mgr.k_pool[layer][block, off : off + n] = k_new[written : written + n]
            self.mgr.v_pool[layer][block, off : off + n] = v_new[written : written + n]
            pos += n
            written += n

        if layer == self.mgr.num_layers - 1:
            self._len[seq_id] = total

        # Gather the whole sequence's K/V through the block table.
        used = (total + bs - 1) // bs
        idx = torch.tensor(table[:used], device=k_new.device)
        k = self.mgr.k_pool[layer][idx].reshape(used * bs, *k_new.shape[1:])[:total]
        v = self.mgr.v_pool[layer][idx].reshape(used * bs, *v_new.shape[1:])[:total]
        return k, v, num_prior
