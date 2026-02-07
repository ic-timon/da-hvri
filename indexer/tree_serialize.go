package indexer

import (
	"bytes"
	"encoding/binary"
	"io"
)

const (
	nodeTagInternal = 0
	nodeTagLeaf     = 1
)

// serializeNode writes the node to w in pre-order. Returns the number of blocks written.
func serializeNode(w io.Writer, n Node, blockData *bytes.Buffer, nextBlockID *int) error {
	if n.IsLeaf() {
		leaf := n.(*LeafNode)
		firstBlockID := *nextBlockID
		vpb := leaf.cfg.VectorsPerBlock
		floatsPerBlock := vpb * BlockDim
		for _, b := range leaf.blocks {
			d := b.Data()
			if len(d) >= floatsPerBlock {
				binary.Write(blockData, binary.LittleEndian, d[:floatsPerBlock])
			} else {
				pad := make([]float32, floatsPerBlock)
				copy(pad, d)
				binary.Write(blockData, binary.LittleEndian, pad)
			}
			*nextBlockID++
		}
		blockCount := len(leaf.blocks)
		// tag
		if err := binary.Write(w, binary.LittleEndian, uint8(nodeTagLeaf)); err != nil {
			return err
		}
		// centroid
		if err := binary.Write(w, binary.LittleEndian, leaf.centroid); err != nil {
			return err
		}
		// vector_count, block_count, first_block_id
		if err := binary.Write(w, binary.LittleEndian, uint32(leaf.vectorCount)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(blockCount)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(firstBlockID)); err != nil {
			return err
		}
		// ids
		if err := binary.Write(w, binary.LittleEndian, leaf.ids); err != nil {
			return err
		}
		return nil
	}
	internal := n.(*InternalNode)
	// tag
	if err := binary.Write(w, binary.LittleEndian, uint8(nodeTagInternal)); err != nil {
		return err
	}
	nc := len(internal.children)
	if err := binary.Write(w, binary.LittleEndian, uint16(nc)); err != nil {
		return err
	}
	for i := 0; i < nc; i++ {
		if err := binary.Write(w, binary.LittleEndian, internal.centroids[i]); err != nil {
			return err
		}
	}
	for i := 0; i < nc; i++ {
		child := internal.Child(i)
		if child != nil {
			if err := serializeNode(w, child, blockData, nextBlockID); err != nil {
				return err
			}
		}
	}
	return nil
}
