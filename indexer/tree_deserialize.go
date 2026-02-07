package indexer

import (
	"bytes"
	"encoding/binary"
	"io"

	"github.com/ic-timon/da-hvri/indexer/store"
)

// deserializeNode reads a node from r and reconstructs it using blockStore and routingOffsets.
func deserializeNode(r io.Reader, cfg *Config, blockStore store.BlockStore, routingOffsets []int64) (Node, error) {
	var tag uint8
	if err := binary.Read(r, binary.LittleEndian, &tag); err != nil {
		return nil, err
	}
	if tag == nodeTagLeaf {
		var centroid [BlockDim]float32
		if err := binary.Read(r, binary.LittleEndian, &centroid); err != nil {
			return nil, err
		}
		var vectorCount, blockCount, firstBlockID uint32
		if err := binary.Read(r, binary.LittleEndian, &vectorCount); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &blockCount); err != nil {
			return nil, err
		}
		if err := binary.Read(r, binary.LittleEndian, &firstBlockID); err != nil {
			return nil, err
		}
		ids := make([]uint64, vectorCount)
		if err := binary.Read(r, binary.LittleEndian, &ids); err != nil {
			return nil, err
		}
		vpb := cfg.VectorsPerBlock
		if vpb <= 0 {
			vpb = 64
		}
		centroidCopy := make([]float32, BlockDim)
		copy(centroidCopy, centroid[:])
		leaf := &LeafNode{
			cfg:         cfg,
			blocks:      make([]Block, 0, blockCount),
			ids:         ids,
			centroid:    centroidCopy,
			vectorCount: int(vectorCount),
		}
		for i := uint32(0); i < blockCount; i++ {
			bid := int(firstBlockID) + int(i)
			if bid >= len(routingOffsets) {
				break
			}
			offset := routingOffsets[bid]
			blk := NewDataBlockMmap(blockStore, offset, vpb)
			leaf.blocks = append(leaf.blocks, blk)
		}
		return leaf, nil
	}
	// internal
	var nc uint16
	if err := binary.Read(r, binary.LittleEndian, &nc); err != nil {
		return nil, err
	}
	internal := NewInternalNode()
	for i := uint16(0); i < nc; i++ {
		var c [BlockDim]float32
		if err := binary.Read(r, binary.LittleEndian, &c); err != nil {
			return nil, err
		}
		internal.centroids = append(internal.centroids, copyVec(c[:]))
	}
	for i := uint16(0); i < nc; i++ {
		child, err := deserializeNode(r, cfg, blockStore, routingOffsets)
		if err != nil {
			return nil, err
		}
		internal.AddChild(child)
	}
	return internal, nil
}

// parseTreeStructure reads the tree structure from data and returns the root node.
func parseTreeStructure(data []byte, cfg *Config, blockStore store.BlockStore, routingOffsets []int64) (Node, error) {
	r := bytes.NewReader(data)
	return deserializeNode(r, cfg, blockStore, routingOffsets)
}
