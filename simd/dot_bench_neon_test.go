//go:build arm64 && cgo

package simd

import (
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

func BenchmarkDotProduct_NEON(b *testing.B) {
	if runtime.GOARCH != "arm64" || !cpu.ARM64.HasNEON {
		b.Skip("NEON not available")
	}
	va, vb := initBenchVectors()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductNEON(va, vb)
	}
}
