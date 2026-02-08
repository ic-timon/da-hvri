package indexer

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/ic-timon/da-hvri/indexer/store"
)

func randomVectors(n int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, BlockDim)
		var norm float64
		for j := 0; j < BlockDim; j++ {
			x := rng.Float32()
			v[j] = x
			norm += float64(x * x)
		}
		norm = math.Sqrt(norm)
		if norm < 1e-9 {
			v[0] = 1
			norm = 1
		}
		for j := 0; j < BlockDim; j++ {
			v[j] /= float32(norm)
		}
		out[i] = v
	}
	return out
}

func TestSearchMultiPathBatch(t *testing.T) {
	cfg := DefaultConfig()
	vecs := randomVectors(100, 42)
	tree := NewTree(cfg)
	for i, v := range vecs {
		tree.Add(v, uint64(i))
	}
	queries := [][]float32{vecs[0], vecs[10], vecs[50]}
	batchResults := tree.SearchMultiPathBatch(queries, 5)
	if len(batchResults) != len(queries) {
		t.Fatalf("batch results count: got %d want %d", len(batchResults), len(queries))
	}
	for i, q := range queries {
		single := tree.SearchMultiPath(q, 5)
		if len(batchResults[i]) != len(single) {
			t.Errorf("query %d: batch len=%d single len=%d", i, len(batchResults[i]), len(single))
		}
		for j := 0; j < len(batchResults[i]) && j < len(single); j++ {
			if batchResults[i][j].ChunkID != single[j].ChunkID {
				t.Errorf("query %d result[%d]: batch ChunkID=%d single=%d", i, j, batchResults[i][j].ChunkID, single[j].ChunkID)
			}
			diff := batchResults[i][j].Score - single[j].Score
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-5 {
				t.Errorf("query %d result[%d]: batch Score=%g single=%g", i, j, batchResults[i][j].Score, single[j].Score)
			}
		}
	}
}

func TestPersist_SerializeDeserializeRoundtrip(t *testing.T) {
	cfg := DefaultConfig()
	cfg.VectorsPerBlock = 64
	cfg.SplitThreshold = 512
	vecs := randomVectors(50, 123)
	pool := NewPool(cfg.VectorsPerBlock)
	leaf := NewLeafNode(pool, cfg)
	for i, v := range vecs {
		leaf.Add(pool, v, uint64(i))
	}
	var treeBuf bytes.Buffer
	var blockBuf bytes.Buffer
	nextBlockID := 0
	if err := serializeNode(&treeBuf, leaf, &blockBuf, &nextBlockID); err != nil {
		t.Fatal(err)
	}

	// Build routing (block i at offset 0, 128KB, 256KB, ...)
	routingOffsets := make([]int64, nextBlockID)
	for i := 0; i < nextBlockID; i++ {
		routingOffsets[i] = int64(i) * int64(store.BlockSizeBytes)
	}
	// For deserialize we need a BlockStore - create a temp file with block data
	tmp := filepath.Join(t.TempDir(), "blocks.bin")
	if err := os.WriteFile(tmp, blockBuf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	blockStore, err := store.OpenMmap(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer blockStore.Close()

	root, err := parseTreeStructure(treeBuf.Bytes(), cfg, blockStore, routingOffsets)
	if err != nil {
		t.Fatalf("parseTreeStructure: %v", err)
	}
	loadedLeaf := root.(*LeafNode)
	if loadedLeaf.VectorCount() != len(vecs) {
		t.Errorf("vectorCount: got %d want %d", loadedLeaf.VectorCount(), len(vecs))
	}
}

func TestPersist_SaveToAtomic(t *testing.T) {
	cfg := DefaultConfig()
	vecs := randomVectors(50, 99)
	tree := NewTree(cfg)
	for i, v := range vecs {
		tree.Add(v, uint64(i))
	}
	tmp := filepath.Join(t.TempDir(), "atomic.bin")
	if err := tree.SaveToAtomic(tmp); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(tmp + ".tmp"); err == nil {
		t.Error("temp file should be removed after rename")
	}
	tree2 := NewTree(cfg)
	if err := tree2.LoadFrom(tmp); err != nil {
		t.Fatal(err)
	}
	defer tree2.ClosePersisted()
	results := tree2.SearchMultiPath(vecs[0], 5)
	if len(results) == 0 {
		t.Error("expected results")
	}
}

func TestNewTreeWithPersistPath(t *testing.T) {
	cfg := DefaultConfig()
	vecs := randomVectors(50, 99)
	tree := NewTree(cfg)
	for i, v := range vecs {
		tree.Add(v, uint64(i))
	}
	tmp := filepath.Join(t.TempDir(), "persistpath.bin")
	if err := tree.SaveToAtomic(tmp); err != nil {
		t.Fatal(err)
	}

	// NewTree with PersistPath should auto LoadFrom (mmap)
	cfg2 := *cfg
	cfg2.PersistPath = tmp
	tree2 := NewTree(&cfg2)
	defer tree2.ClosePersisted()

	if tree2.Pool() != nil {
		t.Error("mmap tree should have nil pool")
	}
	results := tree2.SearchMultiPath(vecs[0], 5)
	if len(results) == 0 {
		t.Error("expected results from PersistPath-loaded tree")
	}
}

func TestNewTreeFromFile(t *testing.T) {
	cfg := DefaultConfig()
	vecs := randomVectors(30, 88)
	tree := NewTree(cfg)
	for i, v := range vecs {
		tree.Add(v, uint64(i))
	}
	tmp := filepath.Join(t.TempDir(), "fromfile.bin")
	if err := tree.SaveToAtomic(tmp); err != nil {
		t.Fatal(err)
	}

	tree2, err := NewTreeFromFile(tmp, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tree2.ClosePersisted()

	results := tree2.SearchMultiPath(vecs[0], 5)
	if len(results) == 0 {
		t.Error("expected results from NewTreeFromFile")
	}
}

func TestPersist_AppendTo(t *testing.T) {
	cfg := DefaultConfig()
	vecs1 := randomVectors(30, 1)
	vecs2 := randomVectors(20, 2)
	tmp := filepath.Join(t.TempDir(), "append.bin")

	// 第一次：新建并追加
	tree, err := AppendTo(tmp, vecs1, nil, cfg)
	if err != nil {
		t.Fatal(err)
	}
	n1 := len(vecs1)
	results := tree.SearchMultiPath(vecs1[0], 5)
	if len(results) == 0 {
		t.Error("expected results after first append")
	}
	tree.ClosePersisted() // 释放 mmap，否则第二次 AppendTo 无法覆盖文件

	// 第二次：从文件加载并追加
	ids := make([]uint64, len(vecs2))
	for i := range ids {
		ids[i] = uint64(n1 + i)
	}
	tree2, err := AppendTo(tmp, vecs2, ids, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tree2.ClosePersisted()
	results2 := tree2.SearchMultiPath(vecs2[0], 5)
	if len(results2) == 0 {
		t.Error("expected results after second append")
	}
}

func TestPersist_SaveLoad_SearchConsistent(t *testing.T) {
	cfg := DefaultConfig()
	cfg.VectorsPerBlock = 64
	cfg.SplitThreshold = 512 // avoid split for simpler test

	vecs := randomVectors(100, 42)
	query := vecs[0]

	tree := NewTree(cfg)
	for i, v := range vecs {
		if !tree.Add(v, uint64(i)) {
			t.Fatalf("Add failed at %d", i)
		}
	}

	resultsBefore := tree.SearchMultiPath(query, 5)
	if len(resultsBefore) == 0 {
		t.Fatal("no results before save")
	}

	t.Run("SaveAndLoad", func(t *testing.T) {
		tmp := filepath.Join(t.TempDir(), "index.bin")
		if err := tree.SaveTo(tmp); err != nil {
			t.Fatalf("SaveTo: %v", err)
		}
		if _, err := os.Stat(tmp); err != nil {
			t.Fatalf("file not created: %v", err)
		}

		tree2 := NewTree(cfg)
		if err := tree2.LoadFrom(tmp); err != nil {
			t.Fatalf("LoadFrom: %v", err)
		}
		defer tree2.ClosePersisted()

		resultsAfter := tree2.SearchMultiPath(query, 5)
		if len(resultsAfter) != len(resultsBefore) {
			t.Errorf("result count: before=%d after=%d", len(resultsBefore), len(resultsAfter))
		}
		for i := range resultsBefore {
			if i >= len(resultsAfter) {
				break
			}
			if resultsBefore[i].ChunkID != resultsAfter[i].ChunkID {
				t.Errorf("result[%d] ChunkID: before=%d after=%d", i, resultsBefore[i].ChunkID, resultsAfter[i].ChunkID)
			}
			diff := resultsBefore[i].Score - resultsAfter[i].Score
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-5 {
				t.Errorf("result[%d] Score: before=%g after=%g", i, resultsBefore[i].Score, resultsAfter[i].Score)
			}
		}
	})
}
