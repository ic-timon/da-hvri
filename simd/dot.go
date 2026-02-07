// Package simd provides AVX-512, AVX2, SSE4, and NEON accelerated vector operations
// for 512-dimensional float32 vectors. Automatically selects the best implementation
// based on GOARCH and CGO availability.
package simd

var (
	dotProductImpl     func(a, b []float32) float64
	dotProductImplDesc string
)

func init() {
	// Default; dispatch files override in init() based on GOARCH and CGO.
	if dotProductImpl == nil {
		dotProductImpl = dotProductGo
		dotProductImplDesc = "Go"
	}
}

// DotProduct computes the dot product of two float32 vectors (cosine similarity for L2-normalized vectors).
// Uses the best available SIMD implementation (AVX-512 > AVX2 > SSE4 on amd64; NEON on arm64).
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	if dotProductImpl != nil {
		return dotProductImpl(a, b)
	}
	return dotProductGo(a, b)
}

// DotProductDesc returns a description of the current dot product implementation (for logging).
func DotProductDesc() string {
	if dotProductImplDesc != "" {
		return dotProductImplDesc
	}
	return "Go"
}

// dotProductGo is the pure Go implementation (4-way unroll, benchmark-optimized).
func dotProductGo(a, b []float32) float64 {
	var sum float64
	for i := 0; i < len(a); i += 4 {
		s0 := a[i+0]*b[i+0] + a[i+1]*b[i+1]
		s1 := a[i+2]*b[i+2] + a[i+3]*b[i+3]
		sum += float64(s0 + s1)
	}
	return sum
}
