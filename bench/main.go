// 压测入口：-stage a|b|c
package main

import (
	"flag"
	"fmt"
	"log"
)

type stageOpts struct {
	shards  int
	offheap bool
}

func main() {
	stage := flag.String("stage", "", "压测阶段: a(参数寻优) | b(容量扩展) | c(高并发) | d(内存vs mmap)")
	shards := flag.Int("shards", 1, "分片数，>1 时使用 ShardedIndex（仅 stage b/c 生效）")
	offheap := flag.Bool("offheap", false, "启用 Off-heap 内存（需 CGO）")
	flag.Parse()
	stageOpts := stageOpts{shards: *shards, offheap: *offheap}
	switch *stage {
	case "a":
		runStageA()
	case "b":
		runStageB(stageOpts)
	case "c":
		runStageC(stageOpts)
	case "d":
		runStageD(stageOpts)
	default:
		log.Fatalf("请指定 -stage a|b|c|d")
	}
	fmt.Println("压测完成")
}
