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

// 模拟 semanticChunk 中约 3365 次点积调用的耗时
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
