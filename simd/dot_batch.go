package simd

const Dim = 512 // Vector dimension (512).

var dotProductBatchFlatImpl func(query []float32, data []float32, n int) []float64

func init() {
	if dotProductBatchFlatImpl == nil {
		dotProductBatchFlatImpl = dotProductBatchFlatGo
	}
}

// DotProductBatchFlat computes dot products of n vectors with query.
// Layout: data[i*Dim:(i+1)*Dim] is the i-th vector. Returns []float64 of length n.
// Uses SIMD when available (AVX-512, AVX2, SSE4, NEON).
func DotProductBatchFlat(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	if dotProductBatchFlatImpl != nil {
		return dotProductBatchFlatImpl(query, data, n)
	}
	return dotProductBatchFlatGo(query, data, n)
}

func dotProductBatchFlatGo(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	for i := 0; i < n; i++ {
		results[i] = DotProduct(query, data[i*Dim:(i+1)*Dim])
	}
	return results
}
