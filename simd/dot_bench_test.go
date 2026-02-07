package simd

import (
	"math/rand"
	"runtime"
	"testing"

	"golang.org/x/sys/cpu"
)

const dim = 512 // BGE 嵌入维度

func initBenchVectors() (va, vb []float32) {
	rand.Seed(42)
	va = make([]float32, dim)
	vb = make([]float32, dim)
	for i := range va {
		va[i] = rand.Float32()*2 - 1
		vb[i] = rand.Float32()*2 - 1
	}
	return va, vb
}

func BenchmarkDotProduct_Go(b *testing.B) {
	va, vb := initBenchVectors()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductGo(va, vb)
	}
}

func BenchmarkDotProduct_AVX512(b *testing.B) {
	va, vb := initBenchVectors()
	if !canUseAVX512() {
		b.Skip("AVX-512 不可用，跳过")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductAVX512(va, vb)
	}
}

func BenchmarkDotProduct_Auto(b *testing.B) {
	va, vb := initBenchVectors()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotProduct(va, vb)
	}
}

// BenchmarkDotProductBatchFlat_Go benchmarks the pure Go batch implementation.
func BenchmarkDotProductBatchFlat_Go(b *testing.B) {
	va, vb := initBenchVectors()
	data := make([]float32, 64*dim)
	for i := 0; i < 64; i++ {
		copy(data[i*dim:(i+1)*dim], va)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotProductBatchFlatGo(vb, data, 64)
	}
}

// BenchmarkDotProductBatchFlat_Auto benchmarks the auto-dispatched batch implementation.
func BenchmarkDotProductBatchFlat_Auto(b *testing.B) {
	va, vb := initBenchVectors()
	data := make([]float32, 64*dim)
	for i := 0; i < 64; i++ {
		copy(data[i*dim:(i+1)*dim], va)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = DotProductBatchFlat(vb, data, 64)
	}
}

// BenchmarkSemanticChunkSim_Go benchmarks ~3365 dot products (simulated semantic chunk).
func BenchmarkSemanticChunkSim_Go(b *testing.B) {
	va, vb := initBenchVectors()
	n := 3365 // 红楼梦段落数
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			_ = dotProductGo(va, vb)
		}
	}
}

func BenchmarkSemanticChunkSim_AVX512(b *testing.B) {
	va, vb := initBenchVectors()
	if !canUseAVX512() {
		b.Skip("AVX-512 不可用，跳过")
	}
	n := 3365
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			_ = dotProductAVX512(va, vb)
		}
	}
}

func canUseAVX512() bool {
	return runtime.GOARCH == "amd64" && cpu.X86.HasAVX512F
}
