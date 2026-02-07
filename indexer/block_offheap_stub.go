//go:build !cgo

package indexer

// allocBlockOffheap returns nil when CGO is disabled, falling back to heap blocks.
func allocBlockOffheap(vectorsPerBlock int) Block {
	return nil
}
