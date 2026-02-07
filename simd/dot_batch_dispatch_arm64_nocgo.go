//go:build arm64 && !cgo

package simd

func init() {
	dotProductBatchFlatImpl = dotProductBatchFlatGo
}
