// Package simd provides AVX-512 accelerated vector operations.
package simd

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// DotProduct computes the dot product of two float32 vectors (cosine similarity for L2-normalized vectors).
// Uses AVX-512 when available, otherwise falls back to pure Go.
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	n := len(a)
	if runtime.GOARCH == "amd64" && cpu.X86.HasAVX512F && n >= 16 {
		return dotProductAVX512(a, b)
	}
	return dotProductGo(a, b)
}

// DotProductDesc returns a description of the current dot product implementation (for logging).
func DotProductDesc() string {
	if runtime.GOARCH == "amd64" && cpu.X86.HasAVX512F {
		return "AVX-512"
	}
	return "Go"
}

// dotProductGo is the pure Go implementation.
func dotProductGo(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i] * b[i])
	}
	return sum
}
