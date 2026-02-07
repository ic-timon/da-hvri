//go:build !amd64 || !cgo

package simd

const Dim = 512 // Vector dimension (512).

// DotProductBatchFlat computes dot products of n vectors with query. Layout: data[i*Dim:(i+1)*Dim] is the i-th vector. Returns []float64 of length n.
func DotProductBatchFlat(query []float32, data []float32, n int) []float64 {
	if len(query) != Dim || n <= 0 || len(data) < n*Dim {
		return nil
	}
	results := make([]float64, n)
	for i := 0; i < n; i++ {
		results[i] = DotProduct(query, data[i*Dim:(i+1)*Dim])
	}
	return results
}
