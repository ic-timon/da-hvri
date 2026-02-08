//go:build arm64 && cgo

package simd

import "golang.org/x/sys/cpu"

func init() {
	if cpu.ARM64.HasASIMD {
		dotProductImpl = dotProductNEON
		dotProductImplDesc = "NEON"
	} else {
		dotProductImpl = dotProductGo
		dotProductImplDesc = "Go"
	}
}
