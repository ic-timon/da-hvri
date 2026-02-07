package store

// BlockStore provides read-only access to persisted blocks.
type BlockStore interface {
	// BlockView returns a []float32 view of the block at the given file offset.
	// The slice is valid until Close is called. Caller must not modify it.
	BlockView(offset int64) []float32
	// Bytes returns the full mapped file as []byte, or nil if not available.
	Bytes() []byte
	// Close releases resources (e.g. unmaps the file).
	Close() error
}
