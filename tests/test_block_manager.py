"""Week 6 unit tests: allocation, ref counting, copy-on-write."""

import pytest
import torch

from mini_engine.paged.block_manager import BlockManager, OutOfBlocks, PagedKVCache


def make_mgr(num_blocks=8, block_size=4):
    return BlockManager(
        num_blocks=num_blocks, block_size=block_size,
        num_layers=2, num_kv_heads=1, head_dim=2,
    )


def test_allocate_free_cycle():
    mgr = make_mgr(num_blocks=4)
    blocks = [mgr.allocate() for _ in range(4)]
    assert mgr.num_free == 0
    with pytest.raises(OutOfBlocks):
        mgr.allocate()
    for b in blocks:
        mgr.decref(b)
    assert mgr.num_free == 4


def test_ref_counting_delays_free():
    mgr = make_mgr()
    b = mgr.allocate()
    mgr.incref(b)  # second owner
    mgr.decref(b)
    assert mgr.ref_count[b] == 1 and b not in mgr.free_list
    mgr.decref(b)
    assert mgr.num_free == mgr.num_blocks


def test_paged_write_read_roundtrip():
    mgr = make_mgr(num_blocks=8, block_size=4)
    kv = PagedKVCache(mgr)
    table = [mgr.allocate() for _ in range(3)]  # room for 12 tokens
    kv.add_sequence(0, table, num_cached_tokens=0)

    k1 = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
    for layer in range(2):
        k, v, prior = kv.append_and_fetch(layer, 0, k1, k1 * 10)
        assert prior == 0
        assert torch.equal(k, k1)
    assert kv.seq_len(0) == 5

    k2 = torch.arange(100, 106, dtype=torch.float32).reshape(3, 1, 2)
    for layer in range(2):
        k, v, prior = kv.append_and_fetch(layer, 0, k2, k2 * 10)
        assert prior == 5
        assert torch.equal(k, torch.cat([k1, k2]))
        assert torch.equal(v, torch.cat([k1, k2]) * 10)


def test_copy_on_write_isolates_shared_tail():
    mgr = make_mgr(num_blocks=8, block_size=4)
    kv = PagedKVCache(mgr)
    table = [mgr.allocate()]
    kv.add_sequence(0, table, 0)
    k = torch.ones(2, 1, 2)
    for layer in range(2):
        kv.append_and_fetch(layer, 0, k, k)

    # Fork: sequence 1 shares the same (partial) block.
    for b in table:
        mgr.incref(b)
    kv.add_sequence(1, list(table), 2)

    kv.ensure_writable_tail(1)
    assert kv.block_tables[1][0] != kv.block_tables[0][0], "COW must copy shared tail"
    k_new = torch.full((1, 1, 2), 7.0)
    for layer in range(2):
        kv.append_and_fetch(layer, 1, k_new, k_new)

    # Sequence 0's data untouched.
    k0, _, _ = kv.append_and_fetch(0, 0, torch.zeros(0, 1, 2), torch.zeros(0, 1, 2))
    assert torch.equal(k0, torch.ones(2, 1, 2))
    # Sequence 1 sees shared prefix + its own token.
    k1_read, _, _ = kv.append_and_fetch(0, 1, torch.zeros(0, 1, 2), torch.zeros(0, 1, 2))
    assert torch.equal(k1_read, torch.cat([torch.ones(2, 1, 2), k_new]))
