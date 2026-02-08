package indexer

import (
	"runtime"
	"sync"
)

// Pool is a memory pool that pre-allocates Blocks (heap or off-heap).
type Pool struct {
	mu              sync.Mutex
	blocks          []Block
	vectorsPerBlock int
	UseOffheap      bool // when true and CGO available, use C.malloc
}

// NewPool creates a memory pool. vectorsPerBlock determines vectors per block.
func NewPool(vectorsPerBlock int) *Pool {
	if vectorsPerBlock <= 0 {
		vectorsPerBlock = 64
	}
	p := &Pool{
		blocks:          make([]Block, 0),
		vectorsPerBlock: vectorsPerBlock,
	}
	runtime.SetFinalizer(p, (*Pool).Close)
	return p
}

// AllocBlock allocates a new Block. Uses off-heap when UseOffheap is true (requires CGO).
func (p *Pool) AllocBlock() Block {
	p.mu.Lock()
	defer p.mu.Unlock()
	var b Block
	if p.UseOffheap {
		b = allocBlockOffheap(p.vectorsPerBlock)
	}
	if b == nil {
		b = NewDataBlock(p.vectorsPerBlock)
	}
	p.blocks = append(p.blocks, b)
	return b
}

// BlockCount returns the number of allocated blocks.
func (p *Pool) BlockCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.blocks)
}

// Close releases all off-heap blocks. Call when the pool is no longer needed.
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, b := range p.blocks {
		b.Close()
	}
	p.blocks = nil
	runtime.SetFinalizer(p, nil)
}
