//go:build amd64 && cgo

package simd

import "golang.org/x/sys/cpu"

func init() {
	if cpu.X86.HasAVX512F {
		dotProductImpl = dotProductAVX512
		dotProductImplDesc = "AVX-512"
	} else if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		dotProductImpl = dotProductAVX2
		dotProductImplDesc = "AVX2"
	} else if cpu.X86.HasSSE41 {
		dotProductImpl = dotProductSSE4
		dotProductImplDesc = "SSE4"
	} else {
		dotProductImpl = dotProductGo
		dotProductImplDesc = "Go"
	}
}
