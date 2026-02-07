//go:build arm64 && cgo

package simd

import "golang.org/x/sys/cpu"

func init() {
	if cpu.ARM64.HasNEON {
		dotProductBatchFlatImpl = dotProductBatchFlatNEON
	} else {
		dotProductBatchFlatImpl = dotProductBatchFlatGo
	}
}
