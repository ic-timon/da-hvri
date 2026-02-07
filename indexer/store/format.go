package store

import (
	"bytes"
	"encoding/binary"
	"errors"
)

const (
	// HeaderSize is the fixed header size (4KB aligned).
	HeaderSize = 64

	// Magic identifies a valid DA-HVRI index file.
	Magic = "DHVR"

	// FormatVersion is the current file format version.
	FormatVersion uint16 = 1

	// BlockSizeBytes is 64 vectors * 512 dim * 4 bytes = 131072.
	BlockSizeBytes = 64 * 512 * 4
)

// Header holds the persisted index metadata.
type Header struct {
	Magic           [4]byte
	Version         uint16
	Dim             uint16
	VectorsPerBlock uint32
	BlockSizeBytes  uint32
	NumBlocks       uint32
	TreeLen         uint32
	RoutingOffset   uint64
	DataOffset      uint64
	Reserved        [16]byte // pad to 64 bytes
}

// EncodeHeader writes the header to a byte slice, padded to HeaderSize.
func EncodeHeader(h *Header) ([]byte, error) {
	if h == nil {
		return nil, errors.New("header is nil")
	}
	copy(h.Magic[:], Magic)
	h.Version = FormatVersion
	var w bytes.Buffer
	if err := binary.Write(&w, binary.LittleEndian, h); err != nil {
		return nil, err
	}
	b := w.Bytes()
	if len(b) < HeaderSize {
		padded := make([]byte, HeaderSize)
		copy(padded, b)
		return padded, nil
	}
	return b, nil
}

// DecodeHeader reads the header from src. Returns error if magic/version invalid.
func DecodeHeader(src []byte) (*Header, error) {
	if len(src) < HeaderSize {
		return nil, errors.New("header too short")
	}
	var h Header
	r := bytes.NewReader(src[:HeaderSize])
	if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
		return nil, err
	}
	if string(h.Magic[:]) != Magic {
		return nil, errors.New("invalid magic")
	}
	if h.Version != FormatVersion {
		return nil, errors.New("unsupported format version")
	}
	return &h, nil
}
