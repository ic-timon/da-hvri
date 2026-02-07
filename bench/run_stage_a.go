package main

import (
	"fmt"
	"time"

	"github.com/ic-timon/da-hvri/bench/gen"
	"github.com/ic-timon/da-hvri/bench/metrics"
	"github.com/ic-timon/da-hvri/indexer"
)

func runStageA() {
	const vectorCount = 10_000
	const dim = 512
	const searchRuns = 100
	const topK = 5

	vpbList := []int{32, 64, 128}
	threshList := []int{128, 256, 512, 1024}
	searchWidth := 3

	vecs := gen.RandomVectors(vectorCount+1, dim, 42)
	query := vecs[vectorCount]
	vecs = vecs[:vectorCount]

	var rows []metrics.StageARow
	for _, vpb := range vpbList {
		for _, thresh := range threshList {
			if thresh < vpb {
				continue
			}
			fmt.Printf("阶段 A: VectorsPerBlock=%d SplitThreshold=%d SearchWidth=%d\n", vpb, thresh, searchWidth)

			metrics.GC()
			_ = metrics.Take()

			cfg := &indexer.Config{
				VectorsPerBlock: vpb,
				SplitThreshold:  thresh,
				SearchWidth:     searchWidth,
			}
			tree := indexer.NewTree(cfg)

			t0 := time.Now()
			for i, v := range vecs {
				if !tree.Add(v, uint64(i)) {
					panic("add failed")
				}
			}
			buildDur := time.Since(t0)

			// 搜索延迟统计
			durations := make([]time.Duration, searchRuns)
			for i := 0; i < searchRuns; i++ {
				t1 := time.Now()
				tree.SearchMultiPath(query, topK)
				durations[i] = time.Since(t1)
			}
			stats := metrics.LatencyStatsFromDurations(durations)

			metrics.GC()
			after := metrics.Take()

			rows = append(rows, metrics.StageARow{
				VectorsPerBlock: vpb,
				SplitThreshold:  thresh,
				SearchWidth:     searchWidth,
				VectorCount:     vectorCount,
				BuildDurMs:      float64(buildDur.Nanoseconds()) / 1e6,
				SearchP50Ms:     stats.P50Ms,
				SearchP99Ms:     stats.P99Ms,
				HeapAllocMB:     float64(after.HeapAlloc) / 1024 / 1024,
			})
			fmt.Printf("  Build=%.0fms SearchP50=%.2fms P99=%.2fms Heap=%.1fMB\n",
				rows[len(rows)-1].BuildDurMs, stats.P50Ms, stats.P99Ms, rows[len(rows)-1].HeapAllocMB)
		}
	}

	path := metrics.ReportPath("bench_report_stage_a_")
	if err := metrics.WriteStageACSV(rows, path); err != nil {
		panic(err)
	}
	fmt.Printf("报告已写入 %s\n", path)
}
