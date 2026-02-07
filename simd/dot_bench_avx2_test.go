//go:build amd64 && cgo

package simd

import (
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkDotProduct_AVX2(b *testing.B) {
	if runtime.GOARCH != "amd64" || !cpu.X86.HasAVX2 {
		b.Skip("AVX2 not available")
	}
	va, vb := initBenchVectors()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductAVX2(va, vb)
	}
}
