// Package metrics 提供运行时指标采集
package metrics

import (
	"runtime"
	"runtime/debug"
	"time"
)

// Snapshot 运行时指标快照
type Snapshot struct {
	TS           time.Time
	HeapAlloc    uint64
	HeapSys      uint64
	HeapReleased uint64
	NumGC        uint32
	NumGoroutine int
}

// Take 采集当前运行时指标
func Take() Snapshot {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return Snapshot{
		TS:           time.Now(),
		HeapAlloc:    m.HeapAlloc,
		HeapSys:      m.HeapSys,
		HeapReleased: m.HeapReleased,
		NumGC:        m.NumGC,
		NumGoroutine: runtime.NumGoroutine(),
	}
}

// GC 触发 GC 并释放回 OS
func GC() {
	runtime.GC()
	debug.FreeOSMemory()
}

// Diff 计算两次快照间的分配速率（bytes/s）和 GC 次数差
func Diff(before, after Snapshot) (allocRateBps float64, gcDelta uint32) {
	elapsed := after.TS.Sub(before.TS).Seconds()
	if elapsed <= 0 {
		return 0, 0
	}
	allocDelta := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	if allocDelta < 0 {
		allocDelta = 0
	}
	allocRateBps = float64(allocDelta) / elapsed
	if after.NumGC >= before.NumGC {
		gcDelta = after.NumGC - before.NumGC
	}
	return allocRateBps, gcDelta
}
