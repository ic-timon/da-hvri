// 阶段 D: 对比纯内存 vs mmap 持久化加载的检索性能（单 Tree）
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ic-timon/da-hvri/bench/gen"
	"github.com/ic-timon/da-hvri/bench/metrics"
	"github.com/ic-timon/da-hvri/indexer"
)

func runStageD(opts stageOpts) {
	const vectorCount = 50_000
	const dim = 512
	const topK = 5
	const totalRequests = 1000
	const concurrency = 16
	const runs = 5 // 多轮取平均

	vecs := gen.RandomVectors(vectorCount+totalRequests, dim, 12345)
	queries := vecs[vectorCount : vectorCount+totalRequests]
	vecs = vecs[:vectorCount]

	cfg := indexer.DefaultConfig()
	cfg.UseOffheap = opts.offheap

	// 1. 纯内存：构建后多轮压测取平均
	fmt.Println("阶段 D: 纯内存模式")
	idxMem := indexer.NewTree(cfg)
	for i, v := range vecs {
		idxMem.Add(v, uint64(i))
	}
	var sumQpsMem, sumP50Mem, sumP99Mem float64
	for r := 0; r < runs; r++ {
		t0 := time.Now()
		durationsMem := runSearchTree(idxMem, queries, topK, concurrency)
		elapsedMem := time.Since(t0).Seconds()
		statsMem := metrics.LatencyStatsFromDurations(durationsMem)
		sumQpsMem += float64(totalRequests) / elapsedMem
		sumP50Mem += statsMem.P50Ms
		sumP99Mem += statsMem.P99Ms
	}
	avgQpsMem := sumQpsMem / float64(runs)
	avgP50Mem := sumP50Mem / float64(runs)
	avgP99Mem := sumP99Mem / float64(runs)
	fmt.Printf("  纯内存 QPS=%.0f P50=%.2fms P99=%.2fms (avg of %d runs)\n", avgQpsMem, avgP50Mem, avgP99Mem, runs)

	// 2. mmap：Save -> (PersistPath + NewTree 或 NewTreeFromFile) -> 多轮压测取平均
	fmt.Println("阶段 D: mmap 持久化模式")
	tmpPath := filepath.Join(os.TempDir(), "da-hvri-stage-d-index.bin")
	if err := idxMem.SaveToAtomic(tmpPath); err != nil {
		panic(err)
	}
	defer os.Remove(tmpPath)

	cfgMmap := *cfg
	cfgMmap.PersistPath = tmpPath
	idxMmap := indexer.NewTree(&cfgMmap)
	defer idxMmap.ClosePersisted()

	var sumQpsMmap, sumP50Mmap, sumP99Mmap float64
	for r := 0; r < runs; r++ {
		t1 := time.Now()
		durationsMmap := runSearchTree(idxMmap, queries, topK, concurrency)
		elapsedMmap := time.Since(t1).Seconds()
		statsMmap := metrics.LatencyStatsFromDurations(durationsMmap)
		sumQpsMmap += float64(totalRequests) / elapsedMmap
		sumP50Mmap += statsMmap.P50Ms
		sumP99Mmap += statsMmap.P99Ms
	}
	avgQpsMmap := sumQpsMmap / float64(runs)
	avgP50Mmap := sumP50Mmap / float64(runs)
	avgP99Mmap := sumP99Mmap / float64(runs)
	fmt.Printf("  mmap QPS=%.0f P50=%.2fms P99=%.2fms (avg of %d runs)\n", avgQpsMmap, avgP50Mmap, avgP99Mmap, runs)
	if avgQpsMem > 0 {
		fmt.Printf("  对比: mmap/内存 QPS 比=%.2f\n", avgQpsMmap/avgQpsMem)
	}
}

func runSearchTree(tree *indexer.Tree, queries [][]float32, topK int, concurrency int) []time.Duration {
	totalRequests := len(queries)
	durations := make([]time.Duration, totalRequests)
	reqPerWorker := totalRequests / concurrency
	if reqPerWorker < 1 {
		reqPerWorker = 1
	}
	var wg sync.WaitGroup
	for c := 0; c < concurrency; c++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			base := worker * reqPerWorker
			for i := 0; i < reqPerWorker && base+i < totalRequests; i++ {
				t1 := time.Now()
				tree.SearchMultiPath(queries[base+i], topK)
				durations[base+i] = time.Since(t1)
			}
		}(c)
	}
	wg.Wait()
	return durations
}
