//go:build amd64 && !cgo

package simd

func init() {
	dotProductBatchFlatImpl = dotProductBatchFlatGo
}
