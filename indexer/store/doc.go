// Package store provides the persist file format and mmap-backed block store
// for the indexer. It is used internally by indexer.SaveTo, indexer.LoadFrom,
// and indexer.NewTreeFromFile.
//
// The file format consists of:
//   - Header (64 bytes): magic, version, metadata
//   - Tree structure: serialized node graph
//   - Block data: contiguous float32 vectors (64 vectors × 512 dim × 4 bytes per block)
package store
