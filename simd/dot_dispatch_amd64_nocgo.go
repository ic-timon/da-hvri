//go:build amd64 && !cgo

package simd

func init() {
	dotProductImpl = dotProductGo
	dotProductImplDesc = "Go"
}
