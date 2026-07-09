"""Week 7 unit tests: longest-prefix match, edge splitting, LRU eviction,
ref-count pinning. block_size=4 to keep token lists short."""

from mini_engine.paged.block_manager import BlockManager
from mini_engine.paged.radix_cache import RadixCache


def setup():
    mgr = BlockManager(num_blocks=64, block_size=4,
                       num_layers=1, num_kv_heads=1, head_dim=2)
    return mgr, RadixCache(mgr)


def seq(mgr, tokens):
    """Simulate a computed sequence: allocate blocks covering `tokens`."""
    n = (len(tokens) + mgr.block_size - 1) // mgr.block_size
    return [mgr.allocate() for _ in range(n)]


def release(mgr, blocks):
    for b in blocks:
        mgr.decref(b)


def test_exact_and_partial_match():
    mgr, radix = setup()
    ids = list(range(12))  # 3 full blocks
    blocks = seq(mgr, ids)
    radix.insert(ids, blocks)
    release(mgr, blocks)  # tree now sole owner

    got, matched = radix.match_prefix(ids)
    assert matched == 12 and got == blocks

    got, matched = radix.match_prefix(ids[:8] + [99, 98])
    assert matched == 8 and got == blocks[:2]

    got, matched = radix.match_prefix([99] * 12)
    assert matched == 0 and got == []


def test_partial_block_never_matches():
    mgr, radix = setup()
    ids = list(range(10))  # 2 full blocks + 2 loose tokens
    blocks = seq(mgr, ids)
    radix.insert(ids, blocks)
    # Only 2 blocks (8 tokens) inserted; the partial block was dropped.
    _, matched = radix.match_prefix(ids)
    assert matched == 8


def test_edge_split_on_divergence():
    mgr, radix = setup()
    a = [1, 2, 3, 4, 5, 6, 7, 8]          # 2 blocks
    b = [1, 2, 3, 4, 9, 9, 9, 9]          # shares block 1 only
    blocks_a = seq(mgr, a)
    radix.insert(a, blocks_a)
    blocks_b = [blocks_a[0]] + seq(mgr, b[4:])
    mgr.incref(blocks_a[0])
    radix.insert(b, blocks_b)

    _, m = radix.match_prefix(a)
    assert m == 8
    _, m = radix.match_prefix(b)
    assert m == 8
    got, m = radix.match_prefix([1, 2, 3, 4, 0, 0, 0, 0])
    assert m == 4 and got == [blocks_a[0]]


def test_lru_eviction_frees_unpinned_only():
    mgr, radix = setup()
    old = list(range(0, 8))
    new = list(range(100, 108))
    blocks_old = seq(mgr, old)
    radix.insert(old, blocks_old)
    release(mgr, blocks_old)
    blocks_new = seq(mgr, new)
    radix.insert(new, blocks_new)
    # `new`'s blocks still referenced by a running sequence (never released).

    free_before = mgr.num_free
    radix.evict(2)
    # LRU: `old` evicted first, its blocks physically freed.
    assert mgr.num_free == free_before + 2
    _, m = radix.match_prefix(old)
    assert m == 0
    # `new` evictable from tree too, but memory not freed while pinned.
    radix.evict(10)
    _, m = radix.match_prefix(new)
    assert m == 0
    for b in blocks_new:
        assert mgr.ref_count[b] == 1  # still owned by the running sequence


def test_match_stats():
    mgr, radix = setup()
    ids = list(range(8))
    blocks = seq(mgr, ids)
    radix.insert(ids, blocks)
    radix.match_prefix(ids)
    assert radix.hit_tokens == 8 and radix.query_tokens == 8
