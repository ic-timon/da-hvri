//go:build arm64 && !cgo

package simd

func init() {
	dotProductImpl = dotProductGo
	dotProductImplDesc = "Go"
}
