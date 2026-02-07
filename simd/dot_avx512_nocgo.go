//go:build !amd64 || !cgo

package simd

// dotProductAVX512 falls back to pure Go when not amd64 or CGO is disabled.
func dotProductAVX512(a, b []float32) float64 {
	return dotProductGo(a, b)
}
