package indexer

import (
	"runtime"
	"sync"
)

var seenPool = sync.Pool{
	New: func() interface{} { return make(map[uint64]float64) },
}

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
	seen := seenPool.Get().(map[uint64]float64)
	defer func() {
		clear(seen)
		seenPool.Put(seen)
	}()
	for _, rs := range results {
		for _, r := range rs {
			if existing, ok := seen[r.ChunkID]; !ok || r.Score > existing {
				seen[r.ChunkID] = r.Score
			}
		}
	}
	return topKFromSeen(seen, k)
}
