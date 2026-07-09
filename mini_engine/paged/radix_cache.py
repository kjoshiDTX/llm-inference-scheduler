"""Week 7: prefix caching via a radix tree (SGLang's RadixAttention, simplified).

Block-level sharing (Week 6) makes prefix reuse *possible*; the radix tree
over token-ID sequences makes it *findable*: on a new request, walk the tree
for the longest matching prefix and skip prefill for every matched block.

Simplification vs SGLang: edges are block-aligned (every edge holds a whole
number of 16-token blocks; a sequence's final partial block is never
inserted). This keeps the node->blocks mapping trivial while preserving the
mechanism: longest-prefix match, edge splitting, LRU eviction with ref-count
pinning.

Ref-count protocol: the tree holds one reference to every block it points at.
A running sequence that reuses a block takes an additional reference. Evicting
a node releases the tree's reference; the block is only truly freed when no
running sequence still uses it.
"""

import time

from .block_manager import BlockManager


class RadixNode:
    __slots__ = ("key", "blocks", "children", "parent", "last_access")

    def __init__(self, key: list[int], blocks: list[int], parent):
        self.key = key            # token ids on the edge into this node
        self.blocks = blocks      # physical blocks holding those tokens' KV
        self.children: dict[int, RadixNode] = {}  # first token -> child
        self.parent = parent
        self.last_access = time.monotonic()


class RadixCache:
    def __init__(self, manager: BlockManager):
        self.mgr = manager
        self.block_size = manager.block_size
        self.root = RadixNode([], [], None)
        # stats for /metrics
        self.hit_tokens = 0
        self.query_tokens = 0

    # --- lookup -------------------------------------------------------------

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """Longest block-aligned prefix match.

        Returns (blocks, num_matched_tokens). Caller must incref the blocks it
        actually adopts (via adopt()).
        """
        self.query_tokens += len(token_ids)
        node = self.root
        blocks: list[int] = []
        matched = 0
        rest = token_ids
        while rest:
            child = node.children.get(rest[0])
            if child is None:
                break
            n = _common_prefix_len(child.key, rest)
            n_blocks = n // self.block_size  # only whole blocks count
            if n_blocks == 0:
                break
            blocks.extend(child.blocks[:n_blocks])
            matched += n_blocks * self.block_size
            child.last_access = time.monotonic()
            if n < len(child.key):
                break  # partial edge match: can't descend further
            node = child
            rest = rest[n:]
        self.hit_tokens += matched
        return blocks, matched

    def adopt(self, blocks: list[int]):
        """A sequence takes shared ownership of matched blocks."""
        for b in blocks:
            self.mgr.incref(b)

    # --- insert ---------------------------------------------------------------

    def insert(self, token_ids: list[int], blocks: list[int]):
        """Register a computed sequence's full blocks for future reuse.

        token_ids is truncated to block alignment; blocks must cover exactly
        those tokens. The tree increfs every block it newly references.
        """
        n_full = len(token_ids) // self.block_size
        token_ids = token_ids[: n_full * self.block_size]
        blocks = blocks[:n_full]

        node = self.root
        rest_tokens = token_ids
        rest_blocks = blocks
        while rest_tokens:
            child = node.children.get(rest_tokens[0])
            if child is None:
                new = RadixNode(list(rest_tokens), list(rest_blocks), node)
                for b in rest_blocks:
                    self.mgr.incref(b)
                node.children[rest_tokens[0]] = new
                return
            n = _common_prefix_len(child.key, rest_tokens)
            n_blocks = n // self.block_size
            if n_blocks == 0:
                # Shares <1 block with this edge; distinct child impossible
                # (same first token) — nothing block-aligned to add here.
                return
            if n_blocks * self.block_size < len(child.key):
                self._split(child, n_blocks)
                child = node.children[rest_tokens[0]]
            child.last_access = time.monotonic()
            node = child
            rest_tokens = rest_tokens[n_blocks * self.block_size :]
            rest_blocks = rest_blocks[n_blocks:]

    def _split(self, node: RadixNode, n_blocks: int):
        """Split an edge after n_blocks whole blocks: parent -> upper -> node."""
        cut = n_blocks * self.block_size
        upper = RadixNode(node.key[:cut], node.blocks[:n_blocks], node.parent)
        upper.last_access = node.last_access
        node.parent.children[node.key[0]] = upper
        node.key = node.key[cut:]
        node.blocks = node.blocks[n_blocks:]
        node.parent = upper
        upper.children[node.key[0]] = node

    # --- eviction ---------------------------------------------------------------

    def evict(self, num_blocks_needed: int) -> int:
        """LRU-evict leaf nodes until num_blocks_needed blocks were released
        (or nothing evictable remains). Returns blocks actually released.

        A leaf whose blocks are still used by running sequences (ref > 1) can
        be released from the tree — decref just drops the tree's reference;
        the memory frees when the sequences finish. That is 'eviction' in the
        accounting sense; physically-free blocks may lag.
        """
        released = 0
        while released < num_blocks_needed:
            leaves = [
                n for n in self._all_nodes() if not n.children and n is not self.root
            ]
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_access)
            for b in victim.blocks:
                self.mgr.decref(b)
            released += len(victim.blocks)
            del victim.parent.children[victim.key[0]]
        return released

    def _all_nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    @property
    def num_tree_blocks(self) -> int:
        return sum(len(n.blocks) for n in self._all_nodes())

    def prefix_state(self, max_nodes: int = 512) -> list[dict]:
        """Compact dump for the router: (token-prefix hash chain, block count).
        The router keeps an approximate copy to score workers."""
        out = []
        for n in self._all_nodes():
            if n is self.root:
                continue
            out.append({"tokens": len(n.key), "blocks": len(n.blocks)})
            if len(out) >= max_nodes:
                break
        return out


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
