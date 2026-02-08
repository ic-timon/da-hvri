package indexer

import (
	"runtime"
	"sync"
)

// ShardedIndex shards vectors across multiple Trees; search queries all shards in parallel and merges results.
type ShardedIndex struct {
	shards  []*Tree
	cfg     *Config
	nShards int
	pool    *searchWorkerPool
}

// NewShardedIndex creates a sharded index. nShards is the number of shards.
func NewShardedIndex(cfg *Config, nShards int) *ShardedIndex {
	if nShards <= 0 {
		nShards = 1
	}
	cfg = cfg.OrDefault()
	shards := make([]*Tree, nShards)
	for i := 0; i < nShards; i++ {
		shards[i] = NewTree(cfg)
	}
	nWorkers := max(nShards, runtime.NumCPU()/2)
	bufSize := 64
	return &ShardedIndex{
		shards:  shards,
		cfg:     cfg,
		nShards: nShards,
		pool:    newSearchWorkerPool(nWorkers, bufSize),
	}
}

// Add inserts a vector, routing to shard by chunkID % nShards.
func (s *ShardedIndex) Add(vec []float32, chunkID uint64) bool {
	idx := chunkID % uint64(s.nShards)
	return s.shards[idx].Add(vec, chunkID)
}

// SearchMultiPathBatch runs batch search across all shards in parallel.
func (s *ShardedIndex) SearchMultiPathBatch(queries [][]float32, k int) [][]SearchResult {
	if len(queries) == 0 || k <= 0 {
		return nil
	}
	shardResults := make([][][]SearchResult, s.nShards)
	var wg sync.WaitGroup
	wg.Add(s.nShards)
	for i, shard := range s.shards {
		idx, sh := i, shard
		go func() {
			defer wg.Done()
			shardResults[idx] = sh.SearchMultiPathBatch(queries, k)
		}()
	}
	wg.Wait()
	seen := make(seenSlice, 0, seenBufCap)
	out := make([][]SearchResult, len(queries))
	for q := range queries {
		seen = seen[:0]
		for sh := 0; sh < s.nShards; sh++ {
			for _, r := range shardResults[sh][q] {
				(&seen).upsert(r.ChunkID, r.Score)
			}
		}
		out[q] = topKFromSeen(seen, k)
	}
	return out
}

// SearchMultiPath queries all shards in parallel and merges Top-K results.
func (s *ShardedIndex) SearchMultiPath(query []float32, k int) []SearchResult {
	if len(query) != BlockDim || k <= 0 {
		return nil
	}
	results := make([][]SearchResult, s.nShards)
	var wg sync.WaitGroup
	wg.Add(s.nShards)
	for i, shard := range s.shards {
		s.pool.Submit(searchJob{i, query, shard, k, results, &wg})
	}
	wg.Wait()
	seen := make(seenSlice, 0, seenBufCap)
	for _, rs := range results {
		for _, r := range rs {
			(&seen).upsert(r.ChunkID, r.Score)
		}
	}
	return topKFromSeen(seen, k)
}
