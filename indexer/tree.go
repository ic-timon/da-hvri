package indexer

import (
	"os"
	"sync/atomic"
)

// Tree is a dynamic descending tree supporting single-path search.
type Tree struct {
	cfg            *Config
	pool           *Pool
	root           atomic.Pointer[Node]
	searchPool     *singleTreeSearchPool
	persistedStore interface{ Close() error } // set by LoadFrom, used by ClosePersisted
}

// NewTree creates a tree. Uses default config if cfg is nil.
// If cfg.PersistPath is non-empty and the file exists, loads from file (mmap, read-only).
// Otherwise creates an empty heap tree for Add.
func NewTree(cfg *Config) *Tree {
	cfg = cfg.OrDefault()
	t := &Tree{cfg: cfg}
	if cfg.PersistPath != "" {
		if _, err := os.Stat(cfg.PersistPath); err == nil {
			if err := t.LoadFrom(cfg.PersistPath); err == nil {
				if cfg.SearchPoolWorkers > 0 {
					t.searchPool = newSingleTreeSearchPool(t, cfg.SearchPoolWorkers, 64)
				}
				return t // mmap tree, pool is nil (read-only)
			}
		}
	}
	pool := NewPool(cfg.VectorsPerBlock)
	pool.UseOffheap = cfg.UseOffheap
	t.pool = pool
	if cfg.SearchPoolWorkers > 0 {
		t.searchPool = newSingleTreeSearchPool(t, cfg.SearchPoolWorkers, 64)
	}
	return t
}

// Config returns the current configuration.
func (t *Tree) Config() *Config {
	return t.cfg
}

// Add inserts a vector. chunkID is the external chunk identifier.
// Returns false if tree is read-only (loaded via PersistPath/LoadFrom).
func (t *Tree) Add(vec []float32, chunkID uint64) bool {
	if t.pool == nil || len(vec) != BlockDim {
		return false
	}
	root := t.root.Load()
	if root == nil {
		leaf := NewLeafNode(t.pool, t.cfg)
		if !leaf.Add(t.pool, vec, chunkID) {
			return false
		}
		np := new(Node)
		*np = leaf
		t.root.Store(np)
		return true
	}
	ok, toSplit := t.addToNode(&t.root, *root, vec, chunkID)
	if ok {
		return true
	}
	if toSplit == nil {
		return false
	}
	// 分裂满叶子并替换，然后重试
	internal := SplitLeaf(toSplit, t.pool)
	if internal == nil {
		return false
	}
	if !t.replaceLeaf(toSplit, internal) {
		return false
	}
	newRoot := t.root.Load()
	if newRoot == nil {
		return false
	}
	ok, _ = t.addToNode(&t.root, *newRoot, vec, chunkID)
	return ok
}

func (t *Tree) addToNode(slot *atomic.Pointer[Node], n Node, vec []float32, chunkID uint64) (ok bool, toSplit *LeafNode) {
	if n.IsLeaf() {
		leaf := n.(*LeafNode)
		if leaf.Add(t.pool, vec, chunkID) {
			return true, nil
		}
		if leaf.VectorCount() >= t.cfg.SplitThreshold {
			return false, leaf
		}
		return false, nil
	}
	internal := n.(*InternalNode)
	idx := internal.BestChild(vec)
	if idx < 0 {
		return false, nil
	}
	child := internal.Child(idx)
	if child == nil {
		return false, nil
	}
	return t.addToNode(internal.ChildSlot(idx), child, vec, chunkID)
}

func (t *Tree) replaceLeaf(old *LeafNode, new *InternalNode) bool {
	return t.replaceInSlot(&t.root, old, new)
}

func (t *Tree) replaceInSlot(slot *atomic.Pointer[Node], old *LeafNode, newInternal *InternalNode) bool {
	p := (*slot).Load()
	if p == nil {
		return false
	}
	node := *p
	if leaf, ok := node.(*LeafNode); ok && leaf == old {
		np := new(Node)
		*np = newInternal
		(*slot).Store(np)
		return true
	}
	if internal, ok := node.(*InternalNode); ok {
		for i := range internal.children {
			if t.replaceInSlot(internal.ChildSlot(i), old, newInternal) {
				return true
			}
		}
	}
	return false
}

// Search performs single-path search and returns Top-K results.
// Lowest latency; use SearchMultiPath for higher recall.
func (t *Tree) Search(query []float32, k int) []SearchResult {
	if len(query) != BlockDim || k <= 0 {
		return nil
	}
	root := t.root.Load()
	if root == nil {
		return nil
	}
	return t.searchNode(*root, query, k)
}

func (t *Tree) searchNode(n Node, query []float32, k int) []SearchResult {
	if n.IsLeaf() {
		return n.(*LeafNode).scanAndTopK(query, k, nil)
	}
	internal := n.(*InternalNode)
	idx := internal.BestChild(query)
	if idx < 0 {
		return nil
	}
	child := internal.Child(idx)
	if child == nil {
		return nil
	}
	return t.searchNode(child, query, k)
}

// Root returns the root node slot (for replacement during split).
func (t *Tree) Root() *atomic.Pointer[Node] {
	return &t.root
}

// Pool returns the memory pool.
func (t *Tree) Pool() *Pool {
	return t.pool
}
