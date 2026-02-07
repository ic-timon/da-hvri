// Package indexer provides the density-adaptive hierarchical vector routing index (DA-HVRI).
//
// Quick start:
//
//	cfg := indexer.DefaultConfig()
//	cfg.UseOffheap = true
//	idx := indexer.NewShardedIndex(cfg, 16)
//	idx.Add(vec, chunkID)
//	results := idx.SearchMultiPath(queryVec, k)
//
package indexer
