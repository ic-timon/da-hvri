package main

import (
	"fmt"
	"time"

	"github.com/ic-timon/da-hvri/bench/gen"
	"github.com/ic-timon/da-hvri/bench/metrics"
	"github.com/ic-timon/da-hvri/indexer"
)

type indexerSearcher interface {
	Add(vec []float32, chunkID uint64) bool
	SearchMultiPath(query []float32, k int) []indexer.SearchResult
}

func runStageB(opts stageOpts) {
	const dim = 512
	const searchRuns = 100
	const topK = 5

	scales := []int{10_000, 50_000, 100_000, 200_000}
	cfg := indexer.DefaultConfig()
	cfg.UseOffheap = opts.offheap

	var rows []metrics.StageBRow
	for _, n := range scales {
		fmt.Printf("阶段 B: 向量规模 %d shards=%d offheap=%v\n", n, opts.shards, opts.offheap)

		vecs := gen.RandomVectors(n+1, dim, int64(n))
		query := vecs[n]
		vecs = vecs[:n]

		metrics.GC()
		_ = metrics.Take()

		var idx indexerSearcher
		if opts.shards > 1 {
			idx = indexer.NewShardedIndex(cfg, opts.shards)
		} else {
			idx = indexer.NewTree(cfg)
		}

		t0 := time.Now()
		for i, v := range vecs {
			if !idx.Add(v, uint64(i)) {
				panic("add failed")
			}
		}
		buildDur := time.Since(t0)

		durations := make([]time.Duration, searchRuns)
		for i := 0; i < searchRuns; i++ {
			t1 := time.Now()
			idx.SearchMultiPath(query, topK)
			durations[i] = time.Since(t1)
		}
		stats := metrics.LatencyStatsFromDurations(durations)

		metrics.GC()
		after := metrics.Take()

		rows = append(rows, metrics.StageBRow{
			VectorCount: n,
			BuildDurMs:  float64(buildDur.Nanoseconds()) / 1e6,
			SearchP50Ms: stats.P50Ms,
			SearchP99Ms: stats.P99Ms,
			HeapSysMB:   float64(after.HeapSys) / 1024 / 1024,
		})
		fmt.Printf("  Build=%.0fms SearchP50=%.2fms P99=%.2fms HeapSys=%.1fMB\n",
			rows[len(rows)-1].BuildDurMs, stats.P50Ms, stats.P99Ms, rows[len(rows)-1].HeapSysMB)

	}

	path := metrics.ReportPath("bench_report_stage_b_")
	if opts.shards > 1 {
		path = metrics.ReportPath("bench_report_stage_b_sharded_")
	}
	if err := metrics.WriteStageBCSV(rows, path); err != nil {
		panic(err)
	}
	fmt.Printf("报告已写入 %s\n", path)
}
