//go:build amd64 && cgo

package simd

import "golang.org/x/sys/cpu"

func init() {
	if cpu.X86.HasAVX512F {
		dotProductBatchFlatImpl = dotProductBatchFlatAVX512
	} else if cpu.X86.HasAVX2 {
		dotProductBatchFlatImpl = dotProductBatchFlatAVX2
	} else if cpu.X86.HasSSE41 {
		dotProductBatchFlatImpl = dotProductBatchFlatSSE4
	} else {
		dotProductBatchFlatImpl = dotProductBatchFlatGo
	}
}
