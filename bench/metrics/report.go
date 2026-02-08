// Package metrics 提供运行时指标采集
package metrics

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// LatencyStats 延迟统计
type LatencyStats struct {
	P50Ms float64
	P95Ms float64
	P99Ms float64
	AvgMs float64
	N     int
}

// StageARow 阶段 A 单行数据
type StageARow struct {
	VectorsPerBlock int
	SplitThreshold  int
	SearchWidth     int
	VectorCount     int
	BuildDurMs      float64
	SearchP50Ms     float64
	SearchP99Ms     float64
	HeapAllocMB     float64
}

// StageBRow 阶段 B 单行数据
type StageBRow struct {
	VectorCount int
	BuildDurMs  float64
	SearchP50Ms float64
	SearchP99Ms float64
	HeapSysMB   float64
}

// StageCRow 阶段 C 单行数据
type StageCRow struct {
	Concurrency  int
	VectorCount  int
	QPS          float64
	SearchP50Ms  float64
	SearchP99Ms  float64
	NumGoroutine int
	P99P50Ratio  float64
}

// Percentile 计算切片中第 p 百分位（0-100），输入需已排序
func Percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	idx := int(float64(len(sorted)-1) * p / 100)
	if idx < 0 {
		idx = 0
	}
	return sorted[idx]
}

// LatencyStatsFromDurations 从耗时列表计算 P50/P95/P99
func LatencyStatsFromDurations(durations []time.Duration) LatencyStats {
	if len(durations) == 0 {
		return LatencyStats{}
	}
	ms := make([]float64, len(durations))
	var sum float64
	for i, d := range durations {
		ms[i] = float64(d.Nanoseconds()) / 1e6
		sum += ms[i]
	}
	// 升序排序
	for i := 0; i < len(ms); i++ {
		for j := i + 1; j < len(ms); j++ {
			if ms[j] < ms[i] {
				ms[i], ms[j] = ms[j], ms[i]
			}
		}
	}
	return LatencyStats{
		P50Ms: Percentile(ms, 50),
		P95Ms: Percentile(ms, 95),
		P99Ms: Percentile(ms, 99),
		AvgMs: sum / float64(len(ms)),
		N:     len(ms),
	}
}

// WriteStageACSV 写入阶段 A 报告
func WriteStageACSV(rows []StageARow, path string) error {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"VectorsPerBlock", "SplitThreshold", "SearchWidth", "VectorCount", "BuildDurMs", "SearchP50Ms", "SearchP99Ms", "HeapAllocMB"})
	for _, r := range rows {
		w.Write([]string{
			fmt.Sprintf("%d", r.VectorsPerBlock),
			fmt.Sprintf("%d", r.SplitThreshold),
			fmt.Sprintf("%d", r.SearchWidth),
			fmt.Sprintf("%d", r.VectorCount),
			fmt.Sprintf("%.2f", r.BuildDurMs),
			fmt.Sprintf("%.2f", r.SearchP50Ms),
			fmt.Sprintf("%.2f", r.SearchP99Ms),
			fmt.Sprintf("%.2f", r.HeapAllocMB),
		})
	}
	w.Flush()
	return w.Error()
}

// WriteStageBCSV 写入阶段 B 报告
func WriteStageBCSV(rows []StageBRow, path string) error {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"VectorCount", "BuildDurMs", "SearchP50Ms", "SearchP99Ms", "HeapSysMB"})
	for _, r := range rows {
		w.Write([]string{
			fmt.Sprintf("%d", r.VectorCount),
			fmt.Sprintf("%.2f", r.BuildDurMs),
			fmt.Sprintf("%.2f", r.SearchP50Ms),
			fmt.Sprintf("%.2f", r.SearchP99Ms),
			fmt.Sprintf("%.2f", r.HeapSysMB),
		})
	}
	w.Flush()
	return w.Error()
}

// WriteStageCCSV 写入阶段 C 报告
func WriteStageCCSV(rows []StageCRow, path string) error {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"Concurrency", "VectorCount", "QPS", "SearchP50Ms", "SearchP99Ms", "NumGoroutine", "P99P50Ratio"})
	for _, r := range rows {
		w.Write([]string{
			fmt.Sprintf("%d", r.Concurrency),
			fmt.Sprintf("%d", r.VectorCount),
			fmt.Sprintf("%.2f", r.QPS),
			fmt.Sprintf("%.2f", r.SearchP50Ms),
			fmt.Sprintf("%.2f", r.SearchP99Ms),
			fmt.Sprintf("%d", r.NumGoroutine),
			fmt.Sprintf("%.2f", r.P99P50Ratio),
		})
	}
	w.Flush()
	return w.Error()
}

// ReportDir 报告输出目录
const ReportDir = "report"

// ReportPath 生成 report/ 目录下带日期的报告路径
func ReportPath(prefix string) string {
	return filepath.Join(ReportDir, prefix+time.Now().Format("20060102")+".csv")
}

// WriteJSON 写入 JSON 报告（通用）
func WriteJSON(v interface{}, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
