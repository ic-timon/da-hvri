// Package indexer provides the density-adaptive hierarchical vector routing index (DA-HVRI).
//
// Vectors must be 512-dimensional and L2-normalized. Use ShardedIndex for
// medium-to-large scale; use Tree for single-tree or small scale.
//
// Quick start (build and search):
//
//	cfg := indexer.DefaultConfig()
//	cfg.UseOffheap = true
//	idx := indexer.NewShardedIndex(cfg, 16)
//	idx.Add(vec, chunkID)
//	results := idx.SearchMultiPath(queryVec, k)
//
// Load from file (mmap, recommended for serving). Set cfg.SearchPoolWorkers =
// runtime.NumCPU() for high-concurrency throttling.
//
//	tree, err := indexer.NewTreeFromFile("/path/to/index.bin", cfg)
//	if err != nil { ... }
//	defer tree.ClosePersisted()
//	results := tree.SearchMultiPath(queryVec, k)
//
// Persistence: SaveToAtomic, LoadFrom, AppendTo. See Tree and NewTreeFromFile.
package indexer
