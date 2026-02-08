package indexer

import (
	"github.com/ic-timon/da-hvri/simd"
	"sync/atomic"
)

// Node is the tree node interface.
type Node interface {
	// IsLeaf returns true if this is a leaf node.
	IsLeaf() bool
	// Centroid returns the centroid vector for routing.
	Centroid() []float32
}

// LeafNode is a leaf node holding Blocks, up to SplitThreshold vectors.
type LeafNode struct {
	cfg         *Config
	blocks      []Block
	ids         []uint64
	centroid    []float32
	vectorCount int
}

// NewLeafNode creates an empty leaf node.
func NewLeafNode(pool *Pool, cfg *Config) *LeafNode {
	cfg = cfg.OrDefault()
	maxBlocks := (cfg.SplitThreshold + cfg.VectorsPerBlock - 1) / cfg.VectorsPerBlock
	if maxBlocks < 1 {
		maxBlocks = 1
	}
	return &LeafNode{
		cfg:         cfg,
		blocks:      make([]Block, 0, maxBlocks),
		ids:         make([]uint64, 0, cfg.SplitThreshold),
		centroid:    make([]float32, BlockDim),
		vectorCount: 0,
	}
}

// IsLeaf implements Node.
func (*LeafNode) IsLeaf() bool { return true }

// Centroid implements Node.
func (n *LeafNode) Centroid() []float32 { return n.centroid }

// VectorCount returns the number of vectors in the leaf.
func (n *LeafNode) VectorCount() int { return n.vectorCount }

// Add appends a vector. Returns false if split is required (at SplitThreshold).
func (n *LeafNode) Add(pool *Pool, vec []float32, chunkID uint64) bool {
	vpb := n.cfg.VectorsPerBlock
	thresh := n.cfg.SplitThreshold
	if len(vec) != BlockDim || n.vectorCount >= thresh {
		return false
	}
	blockIdx := n.vectorCount / vpb
	slot := n.vectorCount % vpb
	if slot == 0 {
		n.blocks = append(n.blocks, pool.AllocBlock())
	}
	n.blocks[blockIdx].SetVector(slot, vec)
	n.ids = append(n.ids, chunkID)
	n.vectorCount++
	n.updateCentroid()
	return true
}

func (n *LeafNode) updateCentroid() {
	if n.vectorCount == 0 {
		return
	}
	vpb := n.cfg.VectorsPerBlock
	for i := 0; i < BlockDim; i++ {
		var sum float32
		for b := 0; b < len(n.blocks); b++ {
			d := n.blocks[b].Data()
			for s := 0; s < vpb && b*vpb+s < n.vectorCount; s++ {
				sum += d[s*BlockDim+i]
			}
		}
		n.centroid[i] = sum / float32(n.vectorCount)
	}
}

// SearchResult holds a single search result returned by Search or SearchMultiPath.
type SearchResult struct {
	ChunkID uint64  // Chunk identifier passed to Add; returned as-is for lookup
	Score   float64 // Cosine similarity (dot product when vectors are L2-normalized)
}

// scanAndTopK 扫描块内向量，返回 Top-K 的 (chunkID, score)
func (n *LeafNode) scanAndTopK(query []float32, k int, bufs *workerBufs) []SearchResult {
	if n.vectorCount == 0 {
		return nil
	}
	vpb := n.cfg.VectorsPerBlock
	var scores []float64
	var indices []int
	if bufs != nil {
		if cap(bufs.scores) < n.vectorCount {
			bufs.scores = make([]float64, n.vectorCount)
		}
		scores = bufs.scores[:n.vectorCount]
		if cap(bufs.indices) < n.vectorCount {
			bufs.indices = make([]int, n.vectorCount)
		}
		indices = bufs.indices[:n.vectorCount]
	} else {
		scores = make([]float64, n.vectorCount)
		indices = make([]int, n.vectorCount)
	}
	offset := 0
	for bi, b := range n.blocks {
		if bi+1 < len(n.blocks) {
			if d := n.blocks[bi+1].Data(); len(d) > 0 {
				_ = d[0]
			}
		}
		if bi+2 < len(n.blocks) {
			if d := n.blocks[bi+2].Data(); len(d) > 0 {
				_ = d[0]
			}
		}
		nInBlock := vpb
		if offset+nInBlock > n.vectorCount {
			nInBlock = n.vectorCount - offset
		}
		batch := b.DotProductBatch(query, nInBlock)
		copy(scores[offset:], batch)
		offset += nInBlock
	}
	return topKFromScores(n.ids, scores, k, indices)
}

// scanAndTopKBatch scans blocks once, computes dot products for all queries, returns one []SearchResult per query.
func (n *LeafNode) scanAndTopKBatch(queries [][]float32, k int, bufs *workerBufs) [][]SearchResult {
	if n.vectorCount == 0 || len(queries) == 0 {
		return nil
	}
	bufs.ensureBatch(len(queries))
	vpb := n.cfg.VectorsPerBlock
	// Ensure each batch score/indices has enough capacity
	for i := 0; i < len(queries); i++ {
		if cap(bufs.batchScores[i]) < n.vectorCount {
			bufs.batchScores[i] = make([]float64, n.vectorCount)
		}
		bufs.batchScores[i] = bufs.batchScores[i][:n.vectorCount]
		if cap(bufs.batchIndices[i]) < n.vectorCount {
			bufs.batchIndices[i] = make([]int, n.vectorCount)
		}
		bufs.batchIndices[i] = bufs.batchIndices[i][:n.vectorCount]
	}
	offset := 0
	for bi, b := range n.blocks {
		if bi+1 < len(n.blocks) {
			if d := n.blocks[bi+1].Data(); len(d) > 0 {
				_ = d[0]
			}
		}
		if bi+2 < len(n.blocks) {
			if d := n.blocks[bi+2].Data(); len(d) > 0 {
				_ = d[0]
			}
		}
		nInBlock := vpb
		if offset+nInBlock > n.vectorCount {
			nInBlock = n.vectorCount - offset
		}
		for q, query := range queries {
			batch := b.DotProductBatch(query, nInBlock)
			copy(bufs.batchScores[q][offset:], batch)
		}
		offset += nInBlock
	}
	out := make([][]SearchResult, len(queries))
	for q := range queries {
		out[q] = topKFromScores(n.ids, bufs.batchScores[q], k, bufs.batchIndices[q])
	}
	return out
}

// InternalNode is an internal node with 2~N children and centroid list.
type InternalNode struct {
	children  []atomic.Pointer[Node]
	centroids [][]float32
}

// NewInternalNode creates an internal node.
func NewInternalNode() *InternalNode {
	return &InternalNode{
		children:  make([]atomic.Pointer[Node], 0),
		centroids: make([][]float32, 0),
	}
}

// IsLeaf implements Node.
func (*InternalNode) IsLeaf() bool { return false }

// Centroid returns the first child's centroid for routing.
func (n *InternalNode) Centroid() []float32 {
	if len(n.centroids) == 0 {
		return nil
	}
	return n.centroids[0]
}

// AddChild adds a child node.
func (n *InternalNode) AddChild(child Node) {
	var p atomic.Pointer[Node]
	np := new(Node)
	*np = child
	p.Store(np)
	n.children = append(n.children, p)
	n.centroids = append(n.centroids, copyVec(child.Centroid()))
}

// BestChild returns the index of the child with highest dot product to query.
func (n *InternalNode) BestChild(query []float32) int {
	if len(n.centroids) == 0 {
		return -1
	}
	best := 0
	bestScore := simd.DotProduct(query, n.centroids[0])
	for i := 1; i < len(n.centroids); i++ {
		s := simd.DotProduct(query, n.centroids[i])
		if s > bestScore {
			bestScore = s
			best = i
		}
	}
	return best
}

// Child returns the i-th child node.
func (n *InternalNode) Child(i int) Node {
	if i < 0 || i >= len(n.children) {
		return nil
	}
	p := n.children[i].Load()
	if p == nil {
		return nil
	}
	return *p
}

// ChildSlot returns the slot for the i-th child (for replaceInSlot).
func (n *InternalNode) ChildSlot(i int) *atomic.Pointer[Node] {
	if i < 0 || i >= len(n.children) {
		return nil
	}
	return &n.children[i]
}

func topKFromScores(ids []uint64, scores []float64, k int, indices []int) []SearchResult {
	if len(ids) != len(scores) || k <= 0 {
		return nil
	}
	if k > len(ids) {
		k = len(ids)
	}
	if indices == nil || len(indices) < len(ids) {
		indices = make([]int, len(ids))
	}
	indices = indices[:len(ids)]
	for i := range indices {
		indices[i] = i
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(indices); j++ {
			if scores[indices[j]] > scores[indices[best]] {
				best = j
			}
		}
		indices[i], indices[best] = indices[best], indices[i]
	}
	out := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		out[i] = SearchResult{ChunkID: ids[indices[i]], Score: scores[indices[i]]}
	}
	return out
}

func copyVec(v []float32) []float32 {
	if v == nil {
		return nil
	}
	o := make([]float32, len(v))
	copy(o, v)
	return o
}
