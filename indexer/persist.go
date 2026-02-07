package indexer

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"

	"github.com/ic-timon/da-hvri/indexer/store"
)

const pageAlign = 4096

func alignUp(x, align int64) int64 {
	if x%align == 0 {
		return x
	}
	return (x/align + 1) * align
}

// NewTreeFromFile loads a tree from file (mmap). cfg may be nil to use DefaultConfig().
// The returned tree is read-only; call ClosePersisted when done.
func NewTreeFromFile(path string, cfg *Config) (*Tree, error) {
	cfg = cfg.OrDefault()
	t := &Tree{cfg: cfg}
	if err := t.LoadFrom(path); err != nil {
		return nil, err
	}
	return t, nil
}

// SaveToAtomic writes the tree to a file atomically (write to path+".tmp", then rename).
// On Windows, the target must not exist for Rename to succeed; remove it first.
func (t *Tree) SaveToAtomic(path string) error {
	tmp := path + ".tmp"
	if err := t.SaveTo(tmp); err != nil {
		return err
	}
	_ = os.Remove(path) // ignore error if not exists
	return os.Rename(tmp, path)
}

// AppendTo loads from path, adds vectors, and saves atomically. Returns the loaded+appended tree.
// If the file does not exist, creates a new tree with the given vectors.
func AppendTo(path string, vecs [][]float32, chunkIDs []uint64, cfg *Config) (*Tree, error) {
	cfg = cfg.OrDefault()
	var t *Tree
	if _, err := os.Stat(path); err == nil {
		loaded, err := NewTreeFromFile(path, cfg)
		if err != nil {
			return nil, err
		}
		// Copy to heap-backed tree (collect before ClosePersisted)
		existingVecs, existingIds := collectVectorsFromNode(loaded.root.Load())
		loaded.ClosePersisted()
		t = NewTree(cfg)
		for i, v := range existingVecs {
			if !t.Add(v, existingIds[i]) {
				return nil, errors.New("add existing vector failed")
			}
		}
	} else {
		t = NewTree(cfg)
	}
	for i, v := range vecs {
		id := uint64(i)
		if i < len(chunkIDs) {
			id = chunkIDs[i]
		}
		if !t.Add(v, id) {
			return nil, errors.New("add new vector failed")
		}
	}
	if err := t.SaveToAtomic(path); err != nil {
		return nil, err
	}
	// Reload from file for mmap-backed read
	t2, err := NewTreeFromFile(path, cfg)
	if err != nil {
		return nil, err
	}
	return t2, nil
}

// collectVectorsFromNode collects all vectors and ids from a tree (for migration from mmap to heap).
func collectVectorsFromNode(n *Node) ([][]float32, []uint64) {
	if n == nil {
		return nil, nil
	}
	node := *n
	if node.IsLeaf() {
		leaf := node.(*LeafNode)
		vecs := make([][]float32, 0, leaf.vectorCount)
		ids := make([]uint64, 0, leaf.vectorCount)
		vpb := leaf.cfg.VectorsPerBlock
		for bi, b := range leaf.blocks {
			d := b.Data()
			if d == nil {
				continue
			}
			nInBlock := vpb
			if (bi+1)*vpb > leaf.vectorCount {
				nInBlock = leaf.vectorCount - bi*vpb
			}
			for s := 0; s < nInBlock; s++ {
				v := make([]float32, BlockDim)
				copy(v, d[s*BlockDim:(s+1)*BlockDim])
				vecs = append(vecs, v)
				ids = append(ids, leaf.ids[bi*vpb+s])
			}
		}
		return vecs, ids
	}
	internal := node.(*InternalNode)
	var allVecs [][]float32
	var allIds []uint64
	for i := 0; i < len(internal.children); i++ {
		child := internal.Child(i)
		if child != nil {
			np := new(Node)
			*np = child
			v, id := collectVectorsFromNode(np)
			allVecs = append(allVecs, v...)
			allIds = append(allIds, id...)
		}
	}
	return allVecs, allIds
}

// SaveTo writes the tree to a file. The tree must not be modified during save.
func (t *Tree) SaveTo(path string) error {
	root := t.root.Load()
	if root == nil {
		return nil
	}
	cfg := t.cfg.OrDefault()

	var treeBuf bytes.Buffer
	var blockBuf bytes.Buffer
	nextBlockID := 0
	if err := serializeNode(&treeBuf, *root, &blockBuf, &nextBlockID); err != nil {
		return err
	}

	treeLen := treeBuf.Len()
	numBlocks := nextBlockID
	blockData := blockBuf.Bytes()
	routingStart := int64(store.HeaderSize) + int64(treeLen)
	dataStart := alignUp(routingStart+int64(numBlocks)*8, pageAlign)

	h := &store.Header{
		Dim:             BlockDim,
		VectorsPerBlock: uint32(cfg.VectorsPerBlock),
		BlockSizeBytes:  store.BlockSizeBytes,
		NumBlocks:       uint32(numBlocks),
		TreeLen:         uint32(treeLen),
		RoutingOffset:   uint64(routingStart),
		DataOffset:      uint64(dataStart),
	}
	headerBytes, err := store.EncodeHeader(h)
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(headerBytes); err != nil {
		return err
	}
	if _, err := f.Write(treeBuf.Bytes()); err != nil {
		return err
	}
	// Routing table
	for i := 0; i < numBlocks; i++ {
		off := dataStart + int64(i)*int64(store.BlockSizeBytes)
		if err := binary.Write(f, binary.LittleEndian, uint64(off)); err != nil {
			return err
		}
	}
	// Pad to dataStart (4KB aligned)
	written := int64(store.HeaderSize) + int64(treeLen) + int64(numBlocks)*8
	padLen := dataStart - written
	if padLen > 0 {
		pad := make([]byte, padLen)
		if _, err := f.Write(pad); err != nil {
			return err
		}
	}
	n, err := f.Write(blockData)
	if err != nil {
		return err
	}
	if n != len(blockData) {
		return errors.New("failed to write full block data")
	}
	// Pad to expected total size (some filesystems may require alignment)
	expectedTotal := dataStart + int64(len(blockData))
	if pos, _ := f.Seek(0, io.SeekCurrent); pos < expectedTotal {
		pad := make([]byte, int(expectedTotal-pos))
		f.Write(pad)
	}
	return f.Sync()
}

// LoadFrom loads a tree from a file. The returned tree is read-only (mmap-backed).
// Caller must call Close on the returned tree's block store when done (via ClosePersisted).
func (t *Tree) LoadFrom(path string) error {
	blockStore, err := store.OpenMmap(path)
	if err != nil {
		return err
	}

	data := blockStore.Bytes()
	if data == nil || len(data) < store.HeaderSize {
		blockStore.Close()
		return errors.New("index file too small or Bytes() not available")
	}

	h, err := store.DecodeHeader(data[:store.HeaderSize])
	if err != nil {
		blockStore.Close()
		return err
	}

	treeStart := int64(store.HeaderSize)
	treeEnd := treeStart + int64(h.TreeLen)
	if int64(len(data)) < treeEnd {
		blockStore.Close()
		return nil
	}
	treeBuf := data[treeStart:treeEnd]

	routingStart := int64(h.RoutingOffset)
	routingEnd := routingStart + int64(h.NumBlocks*8)
	if int64(len(data)) < routingEnd {
		blockStore.Close()
		return nil
	}
	routingBuf := data[routingStart:routingEnd]

	// Ensure we have block data in the file
	dataEnd := int64(h.DataOffset) + int64(h.NumBlocks)*int64(store.BlockSizeBytes)
	if int64(len(data)) < dataEnd {
		blockStore.Close()
		return errors.New("index file truncated")
	}

	routingOffsets := make([]int64, h.NumBlocks)
	r := bytes.NewReader(routingBuf)
	for i := uint32(0); i < h.NumBlocks; i++ {
		var off uint64
		if err := binary.Read(r, binary.LittleEndian, &off); err != nil {
			blockStore.Close()
			return err
		}
		routingOffsets[i] = int64(off)
	}

	cfg := t.cfg.OrDefault()
	cfg.VectorsPerBlock = int(h.VectorsPerBlock)
	if cfg.VectorsPerBlock <= 0 {
		cfg.VectorsPerBlock = 64
	}

	root, err := parseTreeStructure(treeBuf, cfg, blockStore, routingOffsets)
	if err != nil {
		blockStore.Close()
		return err
	}

	np := new(Node)
	*np = root
	t.root.Store(np)
	t.persistedStore = blockStore
	return nil
}

// ClosePersisted releases the mmap for a tree loaded via LoadFrom. No-op if not loaded from file.
func (t *Tree) ClosePersisted() error {
	if t.persistedStore != nil {
		err := t.persistedStore.Close()
		t.persistedStore = nil
		return err
	}
	return nil
}
