package indexer

import (
	"github.com/ic-timon/da-hvri/indexer/store"
	"github.com/ic-timon/da-hvri/simd"
)

// DataBlockMmap is a Block backed by mmap'd file region (read-only).
type DataBlockMmap struct {
	store            store.BlockStore
	offset           int64
	vectorsPerBlock  int
}

// NewDataBlockMmap creates a block view from the store at the given offset.
func NewDataBlockMmap(s store.BlockStore, offset int64, vectorsPerBlock int) *DataBlockMmap {
	if vectorsPerBlock <= 0 {
		vectorsPerBlock = 64
	}
	return &DataBlockMmap{
		store:           s,
		offset:          offset,
		vectorsPerBlock: vectorsPerBlock,
	}
}

// VectorsPerBlock returns the number of vectors in the block.
func (b *DataBlockMmap) VectorsPerBlock() int {
	return b.vectorsPerBlock
}

// FloatsPerBlock returns the number of float32 values in the block.
func (b *DataBlockMmap) FloatsPerBlock() int {
	return b.vectorsPerBlock * BlockDim
}

// Data returns the underlying slice for use with simd.DotProductBatchFlat.
func (b *DataBlockMmap) Data() []float32 {
	return b.store.BlockView(b.offset)
}

// SetVector is a no-op for mmap blocks (read-only).
func (b *DataBlockMmap) SetVector(slot int, vec []float32) {}

// GetVector reads the vector at slot into dst.
func (b *DataBlockMmap) GetVector(slot int, dst []float32) bool {
	if slot < 0 || slot >= b.vectorsPerBlock || len(dst) != BlockDim {
		return false
	}
	d := b.Data()
	if d == nil {
		return false
	}
	start := slot * BlockDim
	copy(dst, d[start:start+BlockDim])
	return true
}

// DotProductBatch computes dot products of the first n vectors with query.
func (b *DataBlockMmap) DotProductBatch(query []float32, n int) []float64 {
	d := b.Data()
	if d == nil {
		return nil
	}
	return simd.DotProductBatchFlat(query, d, n)
}

// Close is a no-op for mmap blocks (store owns the mapping).
func (b *DataBlockMmap) Close() {}
