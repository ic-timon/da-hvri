package indexer

import "github.com/ic-timon/da-hvri/simd"

const (
	// BlockDim is the vector dimension (512).
	BlockDim = 512
)

// Block is the block interface, supporting both heap and off-heap implementations.
type Block interface {
	VectorsPerBlock() int
	FloatsPerBlock() int
	Data() []float32
	SetVector(slot int, vec []float32)
	GetVector(slot int, dst []float32) bool
	DotProductBatch(query []float32, n int) []float64
	Close() // releases resources; no-op for heap blocks, C.free for off-heap
}

// DataBlock stores N vectors of 512 dimensions in heap memory. Layout: [v0_0..v0_511, v1_0..v1_511, ...]
type DataBlock struct {
	data             []float32
	vectorsPerBlock  int
}

// NewDataBlock creates a new block. vectorsPerBlock determines the number of vectors per block.
func NewDataBlock(vectorsPerBlock int) *DataBlock {
	if vectorsPerBlock <= 0 {
		vectorsPerBlock = 64
	}
	n := vectorsPerBlock * BlockDim
	return &DataBlock{
		data:            make([]float32, n),
		vectorsPerBlock: vectorsPerBlock,
	}
}

// VectorsPerBlock returns the number of vectors in the block.
func (b *DataBlock) VectorsPerBlock() int {
	return b.vectorsPerBlock
}

// FloatsPerBlock returns the number of float32 values in the block.
func (b *DataBlock) FloatsPerBlock() int {
	return b.vectorsPerBlock * BlockDim
}

// Data returns the underlying slice for use with simd.DotProductBatchFlat.
func (b *DataBlock) Data() []float32 {
	return b.data
}

// SetVector writes the vector at slot (0-based).
func (b *DataBlock) SetVector(slot int, vec []float32) {
	if slot < 0 || slot >= b.vectorsPerBlock || len(vec) != BlockDim {
		return
	}
	start := slot * BlockDim
	copy(b.data[start:start+BlockDim], vec)
}

// GetVector reads the vector at slot into dst.
func (b *DataBlock) GetVector(slot int, dst []float32) bool {
	if slot < 0 || slot >= b.vectorsPerBlock || len(dst) != BlockDim {
		return false
	}
	start := slot * BlockDim
	copy(dst, b.data[start:start+BlockDim])
	return true
}

// DotProductBatch computes dot products of the first n vectors with query.
func (b *DataBlock) DotProductBatch(query []float32, n int) []float64 {
	return simd.DotProductBatchFlat(query, b.data, n)
}

// Close is a no-op for heap blocks.
func (b *DataBlock) Close() {}
