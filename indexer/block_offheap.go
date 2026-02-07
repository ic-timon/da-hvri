//go:build cgo

package indexer

/*
#include <stdlib.h>
*/
import "C"

import (
	"github.com/ic-timon/da-hvri/simd"
	"unsafe"
)

// DataBlockOffheap is an off-heap block allocated with C.malloc, reducing GC pressure.
type DataBlockOffheap struct {
	ptr             unsafe.Pointer
	vectorsPerBlock int
}

// NewDataBlockOffheap creates an off-heap block. vectorsPerBlock determines vectors per block.
func NewDataBlockOffheap(vectorsPerBlock int) *DataBlockOffheap {
	if vectorsPerBlock <= 0 {
		vectorsPerBlock = 64
	}
	n := vectorsPerBlock * BlockDim
	bytes := n * 4 // sizeof(float32)
	ptr := C.malloc(C.size_t(bytes))
	if ptr == nil {
		return nil
	}
	return &DataBlockOffheap{
		ptr:             unsafe.Pointer(ptr),
		vectorsPerBlock: vectorsPerBlock,
	}
}

// VectorsPerBlock returns the number of vectors in the block.
func (b *DataBlockOffheap) VectorsPerBlock() int {
	return b.vectorsPerBlock
}

// FloatsPerBlock returns the number of float32 values in the block.
func (b *DataBlockOffheap) FloatsPerBlock() int {
	return b.vectorsPerBlock * BlockDim
}

// Data returns a slice view of the off-heap memory.
func (b *DataBlockOffheap) Data() []float32 {
	n := b.vectorsPerBlock * BlockDim
	return unsafe.Slice((*float32)(b.ptr), n)
}

// SetVector writes the vector at slot (0-based).
func (b *DataBlockOffheap) SetVector(slot int, vec []float32) {
	if slot < 0 || slot >= b.vectorsPerBlock || len(vec) != BlockDim {
		return
	}
	start := slot * BlockDim
	d := b.Data()
	copy(d[start:start+BlockDim], vec)
}

// GetVector reads the vector at slot into dst.
func (b *DataBlockOffheap) GetVector(slot int, dst []float32) bool {
	if slot < 0 || slot >= b.vectorsPerBlock || len(dst) != BlockDim {
		return false
	}
	start := slot * BlockDim
	d := b.Data()
	copy(dst, d[start:start+BlockDim])
	return true
}

// DotProductBatch computes dot products of the first n vectors with query.
func (b *DataBlockOffheap) DotProductBatch(query []float32, n int) []float64 {
	return simd.DotProductBatchFlat(query, b.Data(), n)
}

// Close frees the C.malloc-allocated memory.
func (b *DataBlockOffheap) Close() {
	if b.ptr != nil {
		C.free(b.ptr)
		b.ptr = nil
	}
}

// allocBlockOffheap 分配 Off-heap 块（仅 CGO 构建时存在）
func allocBlockOffheap(vectorsPerBlock int) Block {
	return NewDataBlockOffheap(vectorsPerBlock)
}
