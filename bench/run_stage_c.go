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

func runStageC(opts stageOpts) {
	const vectorCount = 50_000
	const dim = 512
	const topK = 5
	const totalRequests = 1000

	concurrencies := []int{1, 4, 8, 16, 32}

	vecs := gen.RandomVectors(vectorCount+totalRequests, dim, 12345)
	queries := vecs[vectorCount : vectorCount+totalRequests]
	vecs = vecs[:vectorCount]

	cfg := indexer.DefaultConfig()
	cfg.UseOffheap = opts.offheap
	useMmap := opts.shards == 1

	var idx indexerSearcher
	if opts.shards > 1 {
		idx = indexer.NewShardedIndex(cfg, opts.shards)
	} else {
		idx = indexer.NewTree(cfg)
	}
	fmt.Printf("阶段 C: 构建 %d 向量索引 shards=%d offheap=%v mmap=%v...\n", vectorCount, opts.shards, opts.offheap, useMmap)
	t0 := time.Now()
	for i, v := range vecs {
		if !idx.Add(v, uint64(i)) {
			panic("add failed")
		}
	}
	fmt.Printf("  构建耗时 %.0fms\n", float64(time.Since(t0).Nanoseconds())/1e6)

	if useMmap {
		// 单树：持久化 -> mmap 加载
		tree := idx.(*indexer.Tree)
		tmpPath := filepath.Join(os.TempDir(), "da-hvri-stage-c-index.bin")
		if err := tree.SaveToAtomic(tmpPath); err != nil {
			panic(err)
		}
		treeMmap, err := indexer.NewTreeFromFile(tmpPath, cfg)
		if err != nil {
			panic(err)
		}
		idx = treeMmap
		defer func() {
			treeMmap.ClosePersisted()
			_ = os.Remove(tmpPath)
		}()
	}

	var rows []metrics.StageCRow
	for _, concurrency := range concurrencies {
		fmt.Printf("阶段 C: 并发数 %d\n", concurrency)

		var wg sync.WaitGroup
		durations := make([]time.Duration, totalRequests)
		reqPerWorker := totalRequests / concurrency
		start := time.Now()
		for c := 0; c < concurrency; c++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				base := worker * reqPerWorker
				for i := 0; i < reqPerWorker && base+i < totalRequests; i++ {
					t1 := time.Now()
					idx.SearchMultiPath(queries[base+i], topK)
					durations[base+i] = time.Since(t1)
				}
			}(c)
		}
		wg.Wait()
		elapsed := time.Since(start).Seconds()

		stats := metrics.LatencyStatsFromDurations(durations)
		qps := float64(totalRequests) / elapsed
		ratio := 1.0
		if stats.P50Ms > 0 {
			ratio = stats.P99Ms / stats.P50Ms
		}

		snap := metrics.Take()
		rows = append(rows, metrics.StageCRow{
			Concurrency:  concurrency,
			VectorCount:  vectorCount,
			QPS:          qps,
			SearchP50Ms:  stats.P50Ms,
			SearchP99Ms:  stats.P99Ms,
			NumGoroutine: snap.NumGoroutine,
			P99P50Ratio:  ratio,
		})
		fmt.Printf("  QPS=%.0f P50=%.2fms P99=%.2fms P99/P50=%.2f Goroutines=%d\n",
			qps, stats.P50Ms, stats.P99Ms, ratio, snap.NumGoroutine)
	}

	path := metrics.ReportPath("bench_report_stage_c_")
	if opts.shards > 1 {
		path = metrics.ReportPath("bench_report_stage_c_sharded_")
	}
	if err := metrics.WriteStageCCSV(rows, path); err != nil {
		panic(err)
	}
	fmt.Printf("报告已写入 %s\n", path)
}
