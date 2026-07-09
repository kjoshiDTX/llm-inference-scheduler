// Week 13: routing policies.
//
// The router cannot see token IDs (it would need the tokenizer), so it keeps
// an approximate, character-level radix tree of the chat prefixes it has sent
// to each worker — the sgl-router design. Characters over-count tokens by
// ~4x uniformly, which cancels out in score *comparisons* between workers.
//
//	score(worker) = alpha * prefixOverlapChars - beta * inflight
//
// alpha rewards steering a conversation back to the worker that already holds
// its KV blocks; beta spreads load when overlap ties (or would create a
// hotspot). round_robin and conversation-ID hashing exist as baselines — the
// final benchmark plots all three.
package main

import (
	"hash/fnv"
	"sync"
	"sync/atomic"
)

// ---- approximate radix tree (character-level) ----

type radixNode struct {
	key      string
	children map[byte]*radixNode
}

func newRadixNode(key string) *radixNode {
	return &radixNode{key: key, children: make(map[byte]*radixNode)}
}

func commonPrefixLen(a, b string) int {
	n := min(len(a), len(b))
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

// PrefixTree tracks what conversation prefixes a worker has (approximately)
// cached. Bounded: prefixes are capped at maxPrefixChars, and the whole tree
// resets past maxNodes (crude but effective eviction at this scale; the
// worker's real radix cache is the ground truth anyway).
type PrefixTree struct {
	mu    sync.Mutex
	root  *radixNode
	nodes int
}

const (
	maxPrefixChars = 8192
	maxNodes       = 4096
)

func NewPrefixTree() *PrefixTree {
	return &PrefixTree{root: newRadixNode("")}
}

func (t *PrefixTree) MatchLen(text string) int {
	if len(text) > maxPrefixChars {
		text = text[:maxPrefixChars]
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	node, matched, rest := t.root, 0, text
	for len(rest) > 0 {
		child, ok := node.children[rest[0]]
		if !ok {
			break
		}
		n := commonPrefixLen(child.key, rest)
		matched += n
		if n < len(child.key) {
			break
		}
		node, rest = child, rest[n:]
	}
	return matched
}

func (t *PrefixTree) Insert(text string) {
	if len(text) > maxPrefixChars {
		text = text[:maxPrefixChars]
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.nodes > maxNodes {
		t.root = newRadixNode("")
		t.nodes = 0
	}
	node, rest := t.root, text
	for len(rest) > 0 {
		child, ok := node.children[rest[0]]
		if !ok {
			node.children[rest[0]] = newRadixNode(rest)
			t.nodes++
			return
		}
		n := commonPrefixLen(child.key, rest)
		if n < len(child.key) {
			// split edge
			upper := newRadixNode(child.key[:n])
			upper.children[child.key[n]] = child
			child.key = child.key[n:]
			node.children[rest[0]] = upper
			t.nodes++
			child = upper
		}
		node, rest = child, rest[n:]
	}
}

// ---- workers and policies ----

type Worker struct {
	ID       string
	BaseURL  string
	Inflight atomic.Int64
	Tree     *PrefixTree
	Requests atomic.Int64
	Healthy  atomic.Bool
}

type Policy interface {
	Pick(workers []*Worker, prefixText string, convID string) *Worker
	Name() string
}

type RoundRobin struct{ next atomic.Uint64 }

func (p *RoundRobin) Name() string { return "round_robin" }
func (p *RoundRobin) Pick(workers []*Worker, _ string, _ string) *Worker {
	for range workers { // skip unhealthy, at most one lap
		w := workers[p.next.Add(1)%uint64(len(workers))]
		if w.Healthy.Load() {
			return w
		}
	}
	return workers[0]
}

// HashAffinity: the old policy — static conversation-ID -> worker mapping.
// Blind spots (measured in the final benchmark): no cross-conversation
// sharing of identical system prompts, no load awareness.
type HashAffinity struct{}

func (p *HashAffinity) Name() string { return "hash" }
func (p *HashAffinity) Pick(workers []*Worker, _ string, convID string) *Worker {
	h := fnv.New64a()
	h.Write([]byte(convID))
	w := workers[h.Sum64()%uint64(len(workers))]
	if !w.Healthy.Load() {
		return (&RoundRobin{}).Pick(workers, "", "")
	}
	return w
}

// CacheAware: score = alpha*overlap - beta*inflight.
type CacheAware struct {
	Alpha float64
	Beta  float64
}

func (p *CacheAware) Name() string { return "cache_aware" }
func (p *CacheAware) Pick(workers []*Worker, prefixText string, _ string) *Worker {
	var best *Worker
	bestScore := 0.0
	for _, w := range workers {
		if !w.Healthy.Load() {
			continue
		}
		overlap := float64(w.Tree.MatchLen(prefixText))
		score := p.Alpha*overlap - p.Beta*float64(w.Inflight.Load())
		// Tie-spreader: with no overlap anywhere and idle workers, all
		// scores are 0 and we'd pile everything onto workers[0]. Nudging by
		// served-request count makes exact ties round-robin.
		score -= 0.01 * float64(w.Requests.Load())
		if best == nil || score > bestScore {
			best, bestScore = w, score
		}
	}
	if best == nil {
		return workers[0]
	}
	return best
}
