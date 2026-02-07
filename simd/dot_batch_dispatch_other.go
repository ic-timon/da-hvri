//go:build !amd64 && !arm64

package simd

func init() {
	dotProductBatchFlatImpl = dotProductBatchFlatGo
}
