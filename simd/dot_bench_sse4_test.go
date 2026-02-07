//go:build amd64 && cgo

package simd

import (
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkDotProduct_SSE4(b *testing.B) {
	if runtime.GOARCH != "amd64" || !cpu.X86.HasSSE41 {
		b.Skip("SSE4.1 not available")
	}
	va, vb := initBenchVectors()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductSSE4(va, vb)
	}
}
