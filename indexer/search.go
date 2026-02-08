package indexer

import (
	"slices"

	"github.com/ic-timon/da-hvri/simd"
)

// SearchMultiPathBatch runs multi-path search for multiple queries in batch.
// Each block is read once and dot products are computed for all queries, improving memory bandwidth utilization.
func (t *Tree) SearchMultiPathBatch(queries [][]float32, k int) [][]SearchResult {
	if len(queries) == 0 || k <= 0 {
		return nil
	}
	for _, q := range queries {
		if len(q) != BlockDim {
			return nil
		}
	}
	root := t.root.Load()
	if root == nil {
		return nil
	}
	sw := t.cfg.SearchWidth
	if sw <= 0 {
		sw = 3
	}
	eps := t.cfg.PruneEpsilon
	candidatesPerLeaf := k * sw
	bufs := newWorkerBufs()
	bufs.ensureBatch(len(queries))
	seenBatch := bufs.seenBatch[:len(queries)]
	// Step 1: collect leaves per query
	leafLists := make([][]*LeafNode, len(queries))
	for i, q := range queries {
		leafLists[i] = t.collectLeaves(*root, q, candidatesPerLeaf, sw, eps, bufs)
	}
	// Step 2: build leaf -> query indices
	leafToQueries := make(map[*LeafNode][]int)
	for i, leaves := range leafLists {
		for _, leaf := range leaves {
			leafToQueries[leaf] = append(leafToQueries[leaf], i)
		}
	}
	// Step 3: batch scan per leaf
	for leaf, qIndices := range leafToQueries {
		batchQueries := make([][]float32, len(qIndices))
		for j, qi := range qIndices {
			batchQueries[j] = queries[qi]
		}
		results := leaf.scanAndTopKBatch(batchQueries, candidatesPerLeaf, bufs)
		for j, qi := range qIndices {
			for _, r := range results[j] {
				(&seenBatch[qi]).upsert(r.ChunkID, r.Score)
			}
		}
	}
	out := make([][]SearchResult, len(queries))
	for i := range queries {
		out[i] = topKFromSeen(seenBatch[i], k)
	}
	return out
}

func (t *Tree) collectLeaves(n Node, query []float32, candidatesPerLeaf int, searchWidth int, pruneEpsilon float64, bufs *workerBufs) []*LeafNode {
	if n.IsLeaf() {
		return []*LeafNode{n.(*LeafNode)}
	}
	internal := n.(*InternalNode)
	indices := topKIndicesWithPruning(internal.centroids, query, searchWidth, pruneEpsilon, bufs)
	var out []*LeafNode
	for _, idx := range indices {
		child := internal.Child(idx)
		if child != nil {
			out = append(out, t.collectLeaves(child, query, candidatesPerLeaf, searchWidth, pruneEpsilon, bufs)...)
		}
	}
	return out
}

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
	return t.searchMultiPathImpl(query, k, nil)
}

// searchMultiPathImpl 内部实现，供 searchPool worker 调用（避免递归进 pool 死锁）
// bufs 为 nil 时退化为 make（兼容无 SearchPool 路径）
func (t *Tree) searchMultiPathImpl(query []float32, k int, bufs *workerBufs) []SearchResult {
	root := t.root.Load()
	if root == nil {
		return nil
	}
	sw := t.cfg.SearchWidth
	if sw <= 0 {
		sw = 3
	}
	eps := t.cfg.PruneEpsilon
	var seen *seenSlice
	if bufs != nil {
		seen = &bufs.seen
	} else {
		s := make(seenSlice, 0, seenBufCap)
		seen = &s
	}
	t.searchMultiPathNode(*root, query, k*sw, sw, eps, seen, bufs)
	return topKFromSeen(*seen, k)
}

func (t *Tree) searchMultiPathNode(n Node, query []float32, candidatesPerLeaf int, searchWidth int, pruneEpsilon float64, seen *seenSlice, bufs *workerBufs) {
	if n.IsLeaf() {
		leaf := n.(*LeafNode)
		results := leaf.scanAndTopK(query, candidatesPerLeaf, bufs)
		for _, r := range results {
			seen.upsert(r.ChunkID, r.Score)
		}
		return
	}
	internal := n.(*InternalNode)
	indices := topKIndicesWithPruning(internal.centroids, query, searchWidth, pruneEpsilon, bufs)
	for _, idx := range indices {
		child := internal.Child(idx)
		if child != nil {
			t.searchMultiPathNode(child, query, candidatesPerLeaf, searchWidth, pruneEpsilon, seen, bufs)
		}
	}
}

// topKIndicesWithPruning 自适应剪枝：仅进入 score >= maxScore - epsilon 的分支，最多 maxK 个
func topKIndicesWithPruning(centroids [][]float32, query []float32, maxK int, epsilon float64, bufs *workerBufs) []int {
	if len(centroids) == 0 || maxK <= 0 {
		return nil
	}
	var scores []float64
	var passed []int
	if bufs != nil {
		if cap(bufs.scores) < len(centroids) {
			bufs.scores = make([]float64, len(centroids))
		}
		scores = bufs.scores[:len(centroids)]
		passed = bufs.indices[:0]
	} else {
		scores = make([]float64, len(centroids))
		passed = make([]int, 0, maxK)
	}
	var dMax float64
	for i, c := range centroids {
		scores[i] = simd.DotProduct(query, c)
		if scores[i] > dMax {
			dMax = scores[i]
		}
	}
	threshold := dMax - epsilon
	for i, s := range scores {
		if s >= threshold {
			passed = append(passed, i)
		}
	}
	if len(passed) <= maxK {
		ret := make([]int, len(passed))
		copy(ret, passed)
		return ret
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
	ret := make([]int, maxK)
	copy(ret, passed[:maxK])
	return ret
}

// topKFromSeen 从 seenSlice 中取 Top-K
func topKFromSeen(seen seenSlice, k int) []SearchResult {
	if len(seen) == 0 || k <= 0 {
		return nil
	}
	slices.SortFunc(seen, func(a, b seenEntry) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return 0
	})
	n := len(seen)
	if n < k {
		k = n
	}
	out := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		out[i] = SearchResult{ChunkID: seen[i].id, Score: seen[i].score}
	}
	return out
}
