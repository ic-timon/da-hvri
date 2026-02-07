package indexer

import (
	"github.com/ic-timon/da-hvri/simd"
	"container/heap"
)

// SearchMultiPath performs multi-path search: at each level selects Top-SearchWidth
// children (or adaptive pruning by PruneEpsilon), deduplicates at leaves, and merges Top-K.
// Higher recall than Search; use for production.
func (t *Tree) SearchMultiPath(query []float32, k int) []SearchResult {
	if len(query) != BlockDim || k <= 0 {
		return nil
	}
	if t.searchPool != nil {
		return t.searchPool.Search(query, k)
	}
	return t.searchMultiPathImpl(query, k)
}

// searchMultiPathImpl 内部实现，供 searchPool worker 调用（避免递归进 pool 死锁）
func (t *Tree) searchMultiPathImpl(query []float32, k int) []SearchResult {
	root := t.root.Load()
	if root == nil {
		return nil
	}
	sw := t.cfg.SearchWidth
	if sw <= 0 {
		sw = 3
	}
	eps := t.cfg.PruneEpsilon
	seen := make(map[uint64]float64) // chunkID -> best score，用于去重
	t.searchMultiPathNode(*root, query, k*sw, sw, eps, seen)
	return topKFromSeen(seen, k)
}

func (t *Tree) searchMultiPathNode(n Node, query []float32, candidatesPerLeaf int, searchWidth int, pruneEpsilon float64, seen map[uint64]float64) {
	if n.IsLeaf() {
		leaf := n.(*LeafNode)
		results := leaf.scanAndTopK(query, candidatesPerLeaf)
		for _, r := range results {
			if existing, ok := seen[r.ChunkID]; !ok || r.Score > existing {
				seen[r.ChunkID] = r.Score
			}
		}
		return
	}
	internal := n.(*InternalNode)
	indices := topKIndicesWithPruning(internal.centroids, query, searchWidth, pruneEpsilon)
	for _, idx := range indices {
		child := internal.Child(idx)
		if child != nil {
			t.searchMultiPathNode(child, query, candidatesPerLeaf, searchWidth, pruneEpsilon, seen)
		}
	}
}

// topKIndicesWithPruning 自适应剪枝：仅进入 score >= maxScore - epsilon 的分支，最多 maxK 个
func topKIndicesWithPruning(centroids [][]float32, query []float32, maxK int, epsilon float64) []int {
	if len(centroids) == 0 || maxK <= 0 {
		return nil
	}
	scores := make([]float64, len(centroids))
	var dMax float64
	for i, c := range centroids {
		scores[i] = simd.DotProduct(query, c)
		if scores[i] > dMax {
			dMax = scores[i]
		}
	}
	threshold := dMax - epsilon
	var passed []int
	for i, s := range scores {
		if s >= threshold {
			passed = append(passed, i)
		}
	}
	if len(passed) <= maxK {
		return passed
	}
	// 超过 maxK 个，取 Top-maxK
	for i := 0; i < maxK; i++ {
		best := i
		for j := i + 1; j < len(passed); j++ {
			if scores[passed[j]] > scores[passed[best]] {
				best = j
			}
		}
		passed[i], passed[best] = passed[best], passed[i]
	}
	return passed[:maxK]
}

// topKFromSeen 从 map[chunkID]score 中取 Top-K
func topKFromSeen(seen map[uint64]float64, k int) []SearchResult {
	if len(seen) == 0 || k <= 0 {
		return nil
	}
	// 使用最小堆保留 Top-K
	h := &minHeap{}
	heap.Init(h)
	for id, score := range seen {
		heap.Push(h, SearchResult{ChunkID: id, Score: score})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	out := make([]SearchResult, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(SearchResult)
	}
	return out
}

// minHeap 最小堆，用于 Top-K
type minHeap []SearchResult

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x interface{}) { *h = append(*h, x.(SearchResult)) }
func (h *minHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
