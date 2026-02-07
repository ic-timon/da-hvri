package store

import (
	"os"
	"unsafe"

	"github.com/edsrzf/mmap-go"
)

// MmapBlockStore is a BlockStore backed by an mmap'd file.
type MmapBlockStore struct {
	f    *os.File
	data mmap.MMap
}

// OpenMmap opens a file and returns a read-only BlockStore.
func OpenMmap(path string) (BlockStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	m, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &MmapBlockStore{f: f, data: m}, nil
}

// Bytes returns the full mapped file.
func (s *MmapBlockStore) Bytes() []byte {
	return s.data
}

// BlockView returns a []float32 view of the block at offset.
// The slice is valid until Close. Caller must not modify it.
func (s *MmapBlockStore) BlockView(offset int64) []float32 {
	if s.data == nil {
		return nil
	}
	if offset < 0 || offset+int64(BlockSizeBytes) > int64(len(s.data)) {
		return nil
	}
	ptr := unsafe.Pointer(&s.data[offset])
	return unsafe.Slice((*float32)(ptr), BlockSizeBytes/4)
}

// Close unmaps the file and closes it.
func (s *MmapBlockStore) Close() error {
	if s.data != nil {
		if err := s.data.Unmap(); err != nil {
			return err
		}
		s.data = nil
	}
	if s.f != nil {
		err := s.f.Close()
		s.f = nil
		return err
	}
	return nil
}
